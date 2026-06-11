"""
Standalone napari viewer for MetaBar results.

Launched as a subprocess from the Streamlit app.
Accepts a JSON config file path as the only argument.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

# ── Patch numba out before napari imports it ──────────────────────────────────
# napari.utils.colormaps.colormap does `import numba` for JIT acceleration.
# In embedded Python, numba/llvmlite fail to load. We replace numba with a
# stub that makes the @numba.jit decorator a no-op.
def _make_numba_stub():
    stub = types.ModuleType("numba")
    def _noop_decorator(*args, **kwargs):
        # Handle both @jit and @jit(...)
        if len(args) == 1 and callable(args[0]):
            return args[0]  # @jit without args
        return lambda f: f  # @jit(...) with args
    stub.jit     = _noop_decorator
    stub.njit    = _noop_decorator
    stub.prange  = range
    stub.float32 = float
    stub.float64 = float
    stub.int32   = int
    stub.int64   = int
    stub.boolean = bool
    stub.types   = types.ModuleType("numba.types")
    stub.core    = types.ModuleType("numba.core")
    # Register submodules
    import sys as _sys
    _sys.modules["numba"]       = stub
    _sys.modules["numba.types"] = stub.types
    _sys.modules["numba.core"]  = stub.core
    return stub

try:
    import numba  # use real numba if available
except ImportError:
    _make_numba_stub()
except Exception:
    _make_numba_stub()
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import napari
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout,
)


# ============================================================
# HELPERS
# ============================================================

def _load_label_mask(path: Path) -> np.ndarray:
    arr = np.squeeze(tiff.imread(str(path)))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got {arr.shape}: {path}")
    return arr.astype(np.int32)


def _load_binary_mask(path: Path) -> np.ndarray:
    arr = np.squeeze(tiff.imread(str(path)))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got {arr.shape}: {path}")
    return arr > 0


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Cannot convert shape {arr.shape} to CxHxW")
    if arr.shape[0] < arr.shape[-1]:
        return arr.astype(np.float32)
    return np.moveaxis(arr, -1, 0).astype(np.float32)


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img)
    mn, mx = img[finite].min(), img[finite].max()
    if mx <= mn:
        return np.zeros_like(img)
    return np.clip((img - mn) / (mx - mn + 1e-8), 0, 1).astype(np.float32)


def _fast_borders(label_mask: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    lab = label_mask.astype(np.int32)
    b = np.zeros(lab.shape, dtype=bool)
    for axis in [0, 1]:
        diff = np.diff(lab, axis=axis) != 0
        nz = (np.take(lab, range(1, lab.shape[axis]), axis=axis) > 0) | \
             (np.take(lab, range(0, lab.shape[axis]-1), axis=axis) > 0)
        diff &= nz
        sl_a = [slice(None)] * 2; sl_a[axis] = slice(1, None)
        sl_b = [slice(None)] * 2; sl_b[axis] = slice(None, -1)
        b[tuple(sl_a)] |= diff
        b[tuple(sl_b)] |= diff
    fg = lab > 0
    b |= fg & (~np.pad(fg[1:, :],  ((0,1),(0,0)), constant_values=False))
    b |= fg & (~np.pad(fg[:-1,:],  ((1,0),(0,0)), constant_values=False))
    b |= fg & (~np.pad(fg[:, 1:],  ((0,0),(0,1)), constant_values=False))
    b |= fg & (~np.pad(fg[:,:-1],  ((0,0),(1,0)), constant_values=False))
    if valid is not None:
        b &= valid
    return b


def _border_rgb(mask: np.ndarray, color: tuple) -> np.ndarray:
    rgb = np.zeros(mask.shape + (3,), dtype=np.float32)
    rgb[mask] = np.array(color, dtype=np.float32)
    return rgb


def _parse_cluster_col(col: str):
    m = re.fullmatch(r"leiden_res_(.+)", col)
    if m:
        return "leiden", m.group(1)
    m = re.fullmatch(r"kmeans_k_(.+)", col)
    if m:
        return "kmeans", m.group(1)
    return None


def _read_color_csv(path: Path) -> dict:
    df = pd.read_csv(str(path))
    return {str(r["cluster"]): (int(r["R"]), int(r["G"]), int(r["B"]))
            for _, r in df.iterrows()}


def _discover_clusters(cluster_dir: Path, name: str, id_col: str) -> dict:
    info = {"available": False, "df": None, "id_col": id_col, "methods": {"leiden": {}, "kmeans": {}, "annotation": {}}}
    csv = cluster_dir / f"{name}__clustered.csv"
    if not csv.exists():
        return info
    df = pd.read_csv(str(csv))
    if id_col not in df.columns:
        return info
    info["df"] = df
    for col in df.columns:
        # First try standard leiden/kmeans pattern
        parsed = _parse_cluster_col(col)
        if parsed is not None:
            method, param = parsed
        else:
            # Fall back: if a colors CSV exists for this column, treat it as an annotation
            color_csv = cluster_dir / f"{col}__colors.csv"
            if color_csv.exists():
                method, param = "annotation", col
            else:
                continue
        color_csv = cluster_dir / f"{col}__colors.csv"
        if not color_csv.exists():
            continue
        if method not in info["methods"]:
            info["methods"][method] = {}
        try:
            info["methods"][method][str(param)] = {
                "col": col,
                "colors": _read_color_csv(color_csv),
            }
        except Exception:
            pass
    if any(info["methods"][m] for m in info["methods"]):
        info["available"] = True
    return info


def _make_cluster_rgb(label_mask, df, id_col, cluster_col, colors, visible_mask=None) -> np.ndarray:
    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    max_id = int(label_mask.max())
    lut = np.zeros((max_id + 1, 3), dtype=np.float32)
    for _, row in df[[id_col, cluster_col]].dropna().iterrows():
        oid = int(row[id_col])
        cl = str(row[cluster_col])
        if 0 <= oid <= max_id and cl in colors:
            lut[oid] = np.array(colors[cl]) / 255.0
    valid = label_mask > 0
    if visible_mask is not None:
        valid &= visible_mask
    rgb[valid] = lut[label_mask[valid]]
    return rgb


def _sorted_channel_lines(row: pd.Series, channels: list[str]) -> list[str]:
    pairs = [(ch, float(row[ch])) for ch in channels if ch in row.index and pd.notna(row[ch])]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [f"    {ch}: {v:.4f}" for ch, v in pairs]


# ============================================================
# UI PANELS
# ============================================================

class InfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selection Info")
        self.resize(600, 700)
        layout = QVBoxLayout()
        self.title = QLabel("Hover or click a cell / superpixel")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        self.lock_label = QLabel("Mode: live hover")
        layout.addWidget(self.lock_label)
        self.clear_btn = QPushButton("Clear locked selection")
        layout.addWidget(self.clear_btn)
        self.setLayout(layout)

    def set_text(self, txt: str):
        self.text.setPlainText(txt)

    def set_locked(self, locked: bool):
        self.lock_label.setText("Mode: locked by click" if locked else "Mode: live hover")


# ============================================================
# BARCODE PANEL
# ============================================================

class BarcodePanel(QWidget):
    """
    Single barcode row for a selected cell:
      MALDI channel mean intensities (viridis, sum-normalised)
    """

    def __init__(self, channel_labels: list):
        super().__init__()
        self.channel_labels = list(channel_labels)
        self.setWindowTitle("Cell Barcode")
        self.resize(900, 200)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.title_label = QLabel("Hover or click a cell")
        layout.addWidget(self.title_label)

        self.fig = Figure(figsize=(11, 4.5))
        self.fig.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas, stretch=1)
        self.setLayout(layout)
        self._draw_empty("Hover or click a cell")

    def _draw_empty(self, msg: str = "No cell selected") -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
                transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def update_cell(self, cell_id: int, maldi_values: np.ndarray,
                    binary_values: np.ndarray = None) -> None:
        self.title_label.setText(f"Cell {cell_id}")
        self.fig.clf()

        ax = self.fig.add_subplot(1, 1, 1)

        maldi = np.nan_to_num(np.asarray(maldi_values, dtype=np.float32), nan=0.0)
        s = maldi.sum()
        if s > 0:
            maldi = maldi / s

        ax.imshow(maldi[np.newaxis, :], aspect="auto", cmap="viridis",
                  interpolation="nearest", vmin=0.0, vmax=float(maldi.max()) or 1.0)
        ax.set_yticks([])

        ax.set_xticks(np.arange(len(self.channel_labels)))
        ax.set_xticklabels(
            self.channel_labels,
            rotation=90, fontsize=6.0, ha="center", va="top",
            fontweight="bold",
        )
        ax.tick_params(axis="x", pad=1)
        ax.set_title(f"Cell {cell_id} — metabolic barcode (sum-normalised)", fontsize=8)

        self.canvas.draw_idle()

    def clear(self, msg: str = "No cell selected") -> None:
        self.title_label.setText(msg)
        self._draw_empty(msg)


class ClusterPanel(QWidget):
    def __init__(self, cell_info: dict, sp_info: dict, ann_info: dict = None, on_change=None):
        super().__init__()
        self.cell_info = cell_info
        self.sp_info   = sp_info
        self.ann_info  = ann_info or {"available": False, "df": None, "id_col": "cell_id", "methods": {"leiden": {}, "kmeans": {}}}
        self.on_change = on_change
        self.setWindowTitle("Cluster Overlays")
        self.resize(380, 300)
        layout = QVBoxLayout()
        self.cell_grp = self._build_group("Cells",       cell_info)
        self.sp_grp   = self._build_group("Superpixels", sp_info)
        self.ann_grp  = self._build_group("Annotations", self.ann_info)
        layout.addWidget(self.cell_grp["box"])
        layout.addWidget(self.sp_grp["box"])
        layout.addWidget(self.ann_grp["box"])
        layout.addStretch(1)
        self.setLayout(layout)
        self._wire(self.cell_grp, cell_info)
        self._wire(self.sp_grp,   sp_info)
        self._wire(self.ann_grp,  self.ann_info)

    def _build_group(self, title: str, info: dict) -> dict:
        box   = QGroupBox(title)
        form  = QFormLayout()
        mc    = QComboBox()
        pc    = QComboBox()
        if info["available"]:
            methods = [m for m in ["leiden", "kmeans", "annotation"] if info["methods"].get(m)]
            mc.addItems(methods)
        else:
            mc.addItem("none"); mc.setEnabled(False)
            pc.addItem("none"); pc.setEnabled(False)
        form.addRow("Method", mc)
        form.addRow("Value",  pc)
        box.setLayout(form)
        return {"box": box, "mc": mc, "pc": pc}

    def _wire(self, grp: dict, info: dict):
        if not info["available"]:
            return
        def refresh():
            m = grp["mc"].currentText()
            if m == "leiden":
                params = sorted(info["methods"].get(m, {}).keys(), key=lambda x: float(x))
            elif m == "kmeans":
                params = sorted(info["methods"].get(m, {}).keys(), key=lambda x: int(float(x)))
            else:
                params = sorted(info["methods"].get(m, {}).keys())
            grp["pc"].blockSignals(True)
            grp["pc"].clear()
            grp["pc"].addItems([str(p) for p in params])
            grp["pc"].blockSignals(False)
        grp["mc"].currentIndexChanged.connect(refresh)
        grp["pc"].currentIndexChanged.connect(self._emit)
        grp["mc"].currentIndexChanged.connect(self._emit)
        refresh()

    def _emit(self):
        if self.on_change:
            self.on_change()

    def state(self) -> dict:
        return {
            "cells":       {"method": self.cell_grp["mc"].currentText(), "param": self.cell_grp["pc"].currentText()},
            "superpixels": {"method": self.sp_grp["mc"].currentText(),   "param": self.sp_grp["pc"].currentText()},
            "annotations": {"method": self.ann_grp["mc"].currentText(),  "param": self.ann_grp["pc"].currentText()},
        }


# ============================================================
# MAIN APP
# ============================================================

class MetaBarcodingViewer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def load_data(self):
        """Load all data — safe to call from a background thread."""
        cfg    = self.cfg
        out_dir = Path(cfg["output_dir"])
        seg_dir = out_dir / "segmentation"
        proj_dir= out_dir / "projection"
        sp_dir  = out_dir / "superpixels"
        cl_dir  = out_dir / "clustering"

        self.channel_labels = cfg["msi_channel_labels"]
        self.if_labels      = cfg.get("if_channel_labels", [])

        print("[LOAD] tables...")
        self.cell_df = pd.read_csv(str(proj_dir / "cell_level_metabolic_table__nuclear_expanded__mean.csv"))
        self.cell_df_idx = self.cell_df.set_index("cell_id", drop=False)

        # Superpixel table is optional
        sp_csv = sp_dir / "mbp_superpixels_mean_intensity_matrix.csv"
        if sp_csv.exists():
            self.sp_df = pd.read_csv(str(sp_csv))
            self.sp_df_idx = self.sp_df.set_index("superpixel_id", drop=False)
        else:
            self.sp_df = pd.DataFrame()
            self.sp_df_idx = pd.DataFrame()
            print("[INFO] No superpixel table found — superpixel features disabled.")

        print("[LOAD] masks...")
        self.cell_mask  = _load_label_mask(seg_dir / "nuclear_mask.tif")
        self.exp_mask   = _load_label_mask(seg_dir / "nuclear_mask_expanded.tif")
        self.nuc_binary = _load_binary_mask(seg_dir / "nuclear_mask_binary.tif")

        # Superpixel mask is optional
        sp_mask_path = sp_dir / "mbp_superpixels_label_mask.tif"
        if sp_mask_path.exists():
            self.sp_mask = _load_label_mask(sp_mask_path)
        else:
            self.sp_mask = np.zeros(self.cell_mask.shape, dtype=np.int32)
            print("[INFO] No superpixel mask found — superpixel overlay disabled.")

        self.visible_mask = self.nuc_binary | (self.sp_mask > 0)
        self.exp_visible  = self.exp_mask > 0

        print("[LOAD] fluorescence stack...")
        if_path = Path(cfg.get("if_stack_path", ""))
        self.if_stack = None
        if if_path.name and if_path.exists():
            self.if_stack = _ensure_chw(tiff.imread(str(if_path)))
            # Auto-read channel names from TIFF metadata if not provided
            if not self.if_labels or len(self.if_labels) != self.if_stack.shape[0]:
                try:
                    with tiff.TiffFile(str(if_path)) as tf:
                        if tf.is_imagej:
                            meta_labels = tf.imagej_metadata.get("Labels", [])
                            if meta_labels:
                                self.if_labels = list(meta_labels)
                except Exception:
                    pass
            if not self.if_labels or len(self.if_labels) != self.if_stack.shape[0]:
                self.if_labels = [f"IF ch {i}" for i in range(self.if_stack.shape[0])]
            print(f"[LOAD] IF stack: {self.if_stack.shape}, labels: {self.if_labels}")

        print("[LOAD] projected MALDI stack (Gaussian)...")
        self.maldi = _ensure_chw(tiff.imread(str(
            proj_dir / "projected_stack_all_channels__full_hr__gaussian.tif")))

        nn_path = proj_dir / "projected_stack_all_channels__full_hr__nearest.tif"
        self.maldi_nn = None
        if nn_path.exists():
            print("[LOAD] projected MALDI stack (nearest-neighbour)...")
            self.maldi_nn = _ensure_chw(tiff.imread(str(nn_path)))

        if self.maldi.shape[0] != len(self.channel_labels):
            raise ValueError(
                f"MALDI stack has {self.maldi.shape[0]} channels but "
                f"{len(self.channel_labels)} labels were provided.")

        print("[BORDER] computing borders...")
        cell_nuc = np.where(self.nuc_binary, self.cell_mask, 0).astype(np.int32)
        self.cell_border_rgb = _border_rgb(_fast_borders(cell_nuc, self.nuc_binary), (0.0, 1.0, 0.0))
        self.exp_border_rgb  = _border_rgb(_fast_borders(self.exp_mask, self.exp_visible), (1.0, 1.0, 0.0))
        self.sp_border_rgb   = _border_rgb(_fast_borders(self.sp_mask, self.sp_mask > 0), (1.0, 1.0, 1.0))

        print("[CLUSTER] discovering results...")
        self.cell_cl = _discover_clusters(cl_dir / "cells",       "cells",       "cell_id")
        self.sp_cl   = _discover_clusters(cl_dir / "superpixels", "superpixels", "superpixel_id")

        # Also discover annotations (saved separately from clustering)
        ann_dir = out_dir / "annotations"
        self.ann_cl = _discover_clusters(ann_dir / "cells", "cells", "cell_id")

        self.threshold_df  = None
        self.binary_df     = None
        self.binary_df_idx = None
        pos_thr_csv = out_dir / "positivity" / "protein_marker_thresholds.csv"
        pos_bin_csv = out_dir / "positivity" / "cell_binary_labels.csv"
        if pos_thr_csv.exists() and pos_bin_csv.exists():
            print("[LOAD] positivity data...")
            try:
                self.threshold_df  = pd.read_csv(str(pos_thr_csv))
                self.binary_df     = pd.read_csv(str(pos_bin_csv))
                self.binary_df_idx = self.binary_df.set_index("cell_id")
                print("[LOAD] positivity data done.")
            except Exception as e:
                print(f"[WARN] Could not load positivity data: {e}")
        else:
            print("[INFO] no positivity data found")

        self.current_ch = self.channel_labels[0]
        self.locked     = False
        self.locked_yx  = None
        print("[LOAD] data loading complete.")

    def build_viewer(self):
        """Create napari window — MUST be called on the main Qt thread."""
        print("[BUILD] creating napari viewer window...")
        self.viewer = napari.Viewer(title="MetaBar Viewer")
        print("[BUILD] adding layers...")
        self._build_layers()
        print("[BUILD] adding widgets...")
        self._build_widgets()
        print("[BUILD] binding callbacks...")
        self._bind_callbacks()
        self.cluster_panel.on_change = self._refresh_clusters
        self._refresh_clusters()
        print("[BUILD] done.")

    # ------------------------------------------------------------------
    def _maldi_img(self, ch: str) -> np.ndarray:
        idx = self.channel_labels.index(ch)
        img = _normalize(self.maldi[idx])
        return np.where(self.visible_mask, img, 0.0).astype(np.float32)

    def _maldi_nn_img(self, ch: str) -> np.ndarray:
        if self.maldi_nn is None:
            return np.zeros(self.cell_mask.shape, dtype=np.float32)
        idx = self.channel_labels.index(ch)
        img = _normalize(self.maldi_nn[idx])
        return np.where(self.visible_mask, img, 0.0).astype(np.float32)

    def _positivity_rgb(self, ch: str) -> np.ndarray:
        """Build positivity overlay for channel ch (red=positive, grey=negative)."""
        h, w = self.exp_mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        if self.binary_df_idx is None:
            return rgb

        col = f"{ch}__positive"
        if col not in self.binary_df.columns:
            return rgb

        max_id = int(self.exp_mask.max())
        lut = np.zeros((max_id + 1, 3), dtype=np.float32)

        # Vectorised — avoid slow iterrows
        ids  = self.binary_df["cell_id"].to_numpy(dtype=np.int32)
        vals = self.binary_df[col].to_numpy(dtype=np.int32)
        valid_mask = (ids >= 0) & (ids <= max_id)
        lut[ids[valid_mask & (vals == 1)]] = (1.0, 0.0, 0.0)
        lut[ids[valid_mask & (vals == 0)]] = (0.25, 0.25, 0.25)

        fg = self.exp_mask > 0
        rgb[fg] = lut[self.exp_mask[fg]]
        return rgb

    # ------------------------------------------------------------------
    def _build_layers(self):
        # IF channels
        self.if_layers = []
        if self.if_stack is not None:
            cmaps = ["red", "green", "blue", "cyan", "magenta", "yellow"]
            for i in range(self.if_stack.shape[0]):
                label = self.if_labels[i] if i < len(self.if_labels) else f"IF ch {i}"
                lyr = self.viewer.add_image(
                    _normalize(self.if_stack[i]),
                    name=f"IF: {label}",
                    opacity=0.8,
                    blending="additive",
                    colormap=cmaps[i % len(cmaps)],
                    visible=True,
                )
                self.if_layers.append(lyr)

        # Projected MALDI (Gaussian)
        self.maldi_layer = self.viewer.add_image(
            self._maldi_img(self.current_ch),
            name=f"MALDI (Gaussian): {self.current_ch}",
            opacity=1.0,
            colormap="viridis",
            blending="translucent",
            visible=False,
        )

        # Projected MALDI (nearest-neighbour / raw)
        self.maldi_nn_layer = self.viewer.add_image(
            self._maldi_nn_img(self.current_ch),
            name=f"MALDI (raw): {self.current_ch}",
            opacity=0.8,
            colormap="magma",
            blending="translucent",
            visible=False,
        )

        # Positivity overlay
        self.pos_layer = self.viewer.add_image(
            self._positivity_rgb(self.current_ch),
            name=f"Positivity: {self.current_ch}",
            rgb=True,
            opacity=0.8,
            blending="translucent",
            visible=False,
        )

        # Cluster overlays
        empty = np.zeros(self.cell_mask.shape + (3,), dtype=np.float32)
        self.sp_cl_layer   = self.viewer.add_image(empty.copy(), name="Superpixel Clusters", rgb=True, opacity=0.8, blending="translucent", visible=False)
        self.cell_cl_layer = self.viewer.add_image(empty.copy(), name="Cell Clusters",       rgb=True, opacity=0.8, blending="translucent", visible=False)
        self.ann_cl_layer  = self.viewer.add_image(empty.copy(), name="Cell Annotations",    rgb=True, opacity=0.8, blending="translucent", visible=False)

        # Borders
        self.sp_border_layer   = self.viewer.add_image(self.sp_border_rgb,   name="Superpixel Borders",    rgb=True, opacity=1.0, blending="additive", visible=False)
        self.cell_border_layer = self.viewer.add_image(self.cell_border_rgb, name="Cell Borders",          rgb=True, opacity=1.0, blending="additive", visible=False)
        self.exp_border_layer  = self.viewer.add_image(self.exp_border_rgb,  name="Expanded Cell Borders", rgb=True, opacity=0.9, blending="additive", visible=False)

    # ------------------------------------------------------------------
    def _build_widgets(self):
        channel_labels = self.channel_labels

        @magicgui(channel={"choices": channel_labels, "label": "MALDI channel"}, auto_call=True)
        def channel_widget(channel=self.current_ch):
            self.current_ch = channel
            self.maldi_layer.data    = self._maldi_img(channel)
            self.maldi_layer.name    = f"MALDI (Gaussian): {channel}"
            self.maldi_nn_layer.data = self._maldi_nn_img(channel)
            self.maldi_nn_layer.name = f"MALDI (raw): {channel}"
            self.pos_layer.data      = self._positivity_rgb(channel)
            self.pos_layer.name      = f"Positivity: {channel}"
            if self.locked and self.locked_yx:
                self._update_info(*self.locked_yx, locked=True)

        self.channel_widget = channel_widget

        self.info_panel    = InfoPanel()
        self.barcode_panel = BarcodePanel(self.channel_labels)
        self.cluster_panel = ClusterPanel(self.cell_cl, self.sp_cl, self.ann_cl)

        self.info_panel.clear_btn.clicked.connect(self._clear_lock)

        self.viewer.window.add_dock_widget(self.info_panel,    area="right", name="Selection Info")
        self.viewer.window.add_dock_widget(self.barcode_panel, area="right", name="Cell Barcode")
        self.viewer.window.add_dock_widget(channel_widget,     area="right", name="MALDI channel")
        self.viewer.window.add_dock_widget(self.cluster_panel, area="right", name="Cluster Overlays")

    # ------------------------------------------------------------------
    def _bind_callbacks(self):
        @self.viewer.mouse_move_callbacks.append
        def _move(viewer, event):
            if self.locked:
                return
            pos = viewer.cursor.position
            if pos and len(pos) >= 2:
                self._update_info(int(round(pos[0])), int(round(pos[1])), locked=False)

        @self.viewer.mouse_drag_callbacks.append
        def _click(viewer, event):
            pos = viewer.cursor.position
            if pos and len(pos) >= 2:
                y, x = int(round(pos[0])), int(round(pos[1]))
                self.locked    = True
                self.locked_yx = (y, x)
                self.info_panel.set_locked(True)
                self._update_info(y, x, locked=True)
            yield

    # ------------------------------------------------------------------
    def _clear_lock(self):
        self.locked    = False
        self.locked_yx = None
        self.info_panel.set_locked(False)
        self.info_panel.set_text("Hover or click a cell / superpixel")
        self.barcode_panel.clear("Hover or click a cell")

    # ------------------------------------------------------------------
    def _refresh_clusters(self):
        state = self.cluster_panel.state()

        for kind, layer, info, mask in [
            ("cells",       self.cell_cl_layer, self.cell_cl, self.exp_mask),
            ("superpixels", self.sp_cl_layer,   self.sp_cl,   self.sp_mask),
            ("annotations", self.ann_cl_layer,  self.ann_cl,  self.exp_mask),
        ]:
            m = state[kind]["method"]
            p = state[kind]["param"]
            if m in ("none", "") or p in ("none", ""):
                continue
            if not info["available"] or m not in info["methods"] or p not in info["methods"][m]:
                continue
            spec = info["methods"][m][p]
            rgb = _make_cluster_rgb(
                mask, info["df"], info["id_col"],
                spec["col"], spec["colors"],
                visible_mask=(mask > 0),
            )
            layer.data = rgb
            layer.name = f"{kind.capitalize()}: {m} {p}"

    # ------------------------------------------------------------------
    def _update_info(self, y: int, x: int, locked: bool):
        H, W = self.cell_mask.shape
        if not (0 <= y < H and 0 <= x < W):
            if not locked:
                self.info_panel.set_text("Outside image")
            return

        cell_id = int(self.cell_mask[y, x])
        exp_id  = int(self.exp_mask[y, x])
        sp_id   = int(self.sp_mask[y, x])
        ch_idx  = self.channel_labels.index(self.current_ch)
        maldi_val = float(self.maldi[ch_idx, y, x])

        lines = [
            f"Pixel: (y={y}, x={x})",
            f"MALDI channel: {self.current_ch}",
            f"MALDI pixel value: {maldi_val:.4f}",
            f"Cell ID: {cell_id if cell_id > 0 else 'None'}",
            f"Expanded Cell ID: {exp_id if exp_id > 0 else 'None'}",
            f"Superpixel ID: {sp_id if sp_id > 0 else 'None'}",
            "",
        ]

        if cell_id > 0 and cell_id in self.cell_df_idx.index:
            row = self.cell_df_idx.loc[cell_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            lines.append("CELL")
            if self.current_ch in row.index:
                lines.append(f"  {self.current_ch}: {row[self.current_ch]:.4f}")
            lines.append("")
            lines.append("  All channels (descending):")
            lines.extend(_sorted_channel_lines(row, self.channel_labels))

            # Update barcode panel
            maldi_vals = np.array(
                [float(row[ch]) if ch in row.index and pd.notna(row[ch]) else 0.0
                 for ch in self.channel_labels], dtype=np.float32
            )
            # Binary positivity row
            binary_vals = np.zeros(len(self.channel_labels), dtype=np.float32)
            if self.binary_df_idx is not None:
                cid_str = str(cell_id)
                if cid_str in self.binary_df_idx.index:
                    brow = self.binary_df_idx.loc[cid_str]
                    if isinstance(brow, pd.DataFrame):
                        brow = brow.iloc[0]
                    for i, ch in enumerate(self.channel_labels):
                        col = f"{ch}__positive"
                        if col in brow.index:
                            try:
                                binary_vals[i] = float(brow[col])
                            except Exception:
                                pass
            self.barcode_panel.update_cell(cell_id, maldi_vals, binary_vals)

        elif sp_id > 0 and sp_id in self.sp_df_idx.index:
            row = self.sp_df_idx.loc[sp_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            lines.append("SUPERPIXEL")
            if self.current_ch in row.index:
                lines.append(f"  {self.current_ch}: {row[self.current_ch]:.4f}")
            lines.append("")
            lines.append("  All channels (descending):")
            lines.extend(_sorted_channel_lines(row, self.channel_labels))
            self.barcode_panel.clear("Superpixel selected — no cell barcode")

        else:
            lines.append("No labeled cell or superpixel at this position.")
            self.barcode_panel.clear("No cell at this position")

        self.info_panel.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    def run(self):
        napari.run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python napari_viewer.py <config.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8-sig") as f:
        cfg = json.load(f)

    import threading

    # Use napari's QApplication — compatible with multiple napari versions
    from qtpy.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QLabel, QMessageBox
    from qtpy.QtCore import Qt, QTimer

    try:
        from napari._qt.qt_event_loop import get_qapp
        qt_app = get_qapp()
    except ImportError:
        try:
            from napari._qt.qt_event_loop import get_app
            qt_app = get_app()
        except ImportError:
            qt_app = QApplication.instance() or QApplication(sys.argv)

    # ---- Splash window ----
    splash_widget = QWidget()
    splash_widget.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash_widget.setFixedSize(420, 140)
    splash_widget.setStyleSheet("background-color: #1e1e2e; border-radius: 8px;")
    layout = QVBoxLayout()
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(12)
    title = QLabel("MetaBar Viewer")
    title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
    title.setAlignment(Qt.AlignCenter)
    status_label = QLabel("Loading data, please wait...")
    status_label.setStyleSheet("color: #aaaacc; font-size: 11px;")
    status_label.setAlignment(Qt.AlignCenter)
    bar = QProgressBar()
    bar.setRange(0, 0)
    bar.setStyleSheet("""
        QProgressBar { border: none; background: #2e2e4e; border-radius: 4px; height: 8px; }
        QProgressBar::chunk { background: #6c63ff; border-radius: 4px; }
    """)
    bar.setTextVisible(False)
    layout.addWidget(title)
    layout.addWidget(status_label)
    layout.addWidget(bar)
    splash_widget.setLayout(layout)
    screen = qt_app.primaryScreen().geometry()
    splash_widget.move(
        screen.center().x() - splash_widget.width() // 2,
        screen.center().y() - splash_widget.height() // 2,
    )
    splash_widget.show()
    qt_app.processEvents()

    # ---- Load data in background thread ----
    viewer_holder = [None]
    error_holder  = [None]
    done_flag     = [False]
    status_holder = ["Initialising..."]

    def _load():
        try:
            import builtins
            _orig = builtins.print
            def _p(*a, **k):
                status_holder[0] = " ".join(str(x) for x in a)
                _orig(*a, **k)
            builtins.print = _p
            v = MetaBarcodingViewer(cfg)
            v.load_data()
            viewer_holder[0] = v
        except Exception:
            import traceback
            error_holder[0] = traceback.format_exc()
        finally:
            done_flag[0] = True

    threading.Thread(target=_load, daemon=True).start()

    elapsed = [0]

    def _check():
        elapsed[0] += 1
        if status_holder[0]:
            status_label.setText(status_holder[0][:80])
        qt_app.processEvents()

        if done_flag[0]:
            timer.stop()
            splash_widget.close()
            if error_holder[0]:
                QMessageBox.critical(None, "MetaBar Viewer Error",
                    f"Failed to load data:\n\n{error_holder[0][:2000]}")
                sys.exit(1)
            try:
                viewer_holder[0].build_viewer()
                napari.run()
            except Exception:
                import traceback
                QMessageBox.critical(None, "MetaBar Viewer Error",
                    f"Failed to build viewer:\n\n{traceback.format_exc()[:2000]}")
                sys.exit(1)

        if elapsed[0] > 1500:
            timer.stop()
            splash_widget.close()
            QMessageBox.critical(None, "Timeout",
                f"Loading timed out.\nLast step: {status_holder[0]}")
            sys.exit(1)

    timer = QTimer()
    timer.timeout.connect(_check)
    timer.start(400)
    napari.run(max_loop_level=2)
