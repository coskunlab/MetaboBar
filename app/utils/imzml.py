"""
imzML extraction utilities: parsing targets, building coordinate maps,
extracting peak intensities, and writing output TIFFs.
"""

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff
from pyimzml.ImzMLParser import ImzMLParser


# ---------------------------------------------------------------------------
# m/z window helpers
# ---------------------------------------------------------------------------

def ppm_window(mz: float, ppm: float) -> Tuple[float, float]:
    delta = mz * ppm * 1e-6
    return mz - delta, mz + delta


def extract_peak_intensity(
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_mz: float,
    ppm: float = 5.0,
) -> float:
    """
    Sum intensities within ±ppm window around target_mz.
    Assumes mzs is sorted ascending.
    """
    lo, hi = ppm_window(target_mz, ppm)
    left = np.searchsorted(mzs, lo, side="left")
    right = np.searchsorted(mzs, hi, side="right")

    if right <= left:
        return 0.0

    return float(np.sum(intensities[left:right]))


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------

def read_targets_from_csv_path(csv_path: Path) -> List[Dict[str, object]]:
    """
    Expects columns: lipid, ion, target_mz
    Also accepts a single m/z column named mz, m/z, or target_mz.
    """
    targets = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")

        fields = set(reader.fieldnames)

        if {"lipid", "ion", "target_mz"}.issubset(fields):
            for row in reader:
                targets.append(
                    {
                        "lipid": str(row["lipid"]).strip(),
                        "ion": str(row["ion"]).strip(),
                        "target_mz": float(row["target_mz"]),
                    }
                )

        elif "target_mz" in fields or "mz" in fields or "m/z" in fields:
            mz_col = (
                "target_mz"
                if "target_mz" in fields
                else ("mz" if "mz" in fields else "m/z")
            )

            for i, row in enumerate(reader, start=1):
                mz = float(row[mz_col])
                targets.append(
                    {
                        "lipid": str(row.get("lipid", f"mz_{i}")).strip(),
                        "ion": str(row.get("ion", "")).strip(),
                        "target_mz": mz,
                    }
                )

        else:
            raise ValueError(
                "CSV must contain lipid, ion, target_mz columns, "
                "or at least one target_mz/mz/m/z column. "
                f"Got columns: {reader.fieldnames}"
            )

    if not targets:
        raise ValueError("No targets found in CSV.")

    return targets


def read_targets_from_text(text: str) -> List[Dict[str, object]]:
    cleaned = text.replace(";", ",").replace("\n", ",").replace("\t", ",")

    if "," not in cleaned:
        cleaned = cleaned.replace(" ", ",")

    values = [v.strip() for v in cleaned.split(",") if v.strip()]

    targets = []
    for i, value in enumerate(values, start=1):
        mz = float(value)
        targets.append({"lipid": f"mz_{i}", "ion": "", "target_mz": mz})

    if not targets:
        raise ValueError("No m/z targets were entered.")

    return targets


# ---------------------------------------------------------------------------
# Coordinate map
# ---------------------------------------------------------------------------

def build_coordinate_maps(
    parser: ImzMLParser,
) -> Tuple[np.ndarray, dict, dict, int, int]:
    coords = np.array(parser.coordinates, dtype=int)

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    return coords, x_to_idx, y_to_idx, len(ys), len(xs)


# ---------------------------------------------------------------------------
# Stack normalisation / dtype conversion
# ---------------------------------------------------------------------------

def normalize_stack(stack: np.ndarray, normalize: Optional[str]) -> np.ndarray:
    if normalize is None:
        return stack

    if normalize == "per_channel_max":
        for c in range(stack.shape[0]):
            m = stack[c].max()
            if m > 0:
                stack[c] /= m
        return stack

    if normalize == "global_max":
        m = stack.max()
        if m > 0:
            stack /= m
        return stack

    raise ValueError("normalize must be None, per_channel_max, or global_max")


def convert_stack_dtype(
    stack: np.ndarray,
    dtype_name: str,
    normalize: Optional[str],
) -> np.ndarray:
    if dtype_name == "uint16":
        if normalize is None:
            return np.clip(stack, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return np.clip(stack * 65535.0, 0, 65535).astype(np.uint16)

    if dtype_name == "uint8":
        if normalize is None:
            return np.clip(stack, 0, np.iinfo(np.uint8).max).astype(np.uint8)
        return np.clip(stack * 255.0, 0, 255).astype(np.uint8)

    return stack.astype(np.float32)


# ---------------------------------------------------------------------------
# Channel name helpers
# ---------------------------------------------------------------------------

def channel_names_from_targets(targets: Sequence[Dict[str, object]]) -> List[str]:
    names = []

    for t in targets:
        lipid = str(t.get("lipid", "")).strip()
        ion = str(t.get("ion", "")).strip()
        mz = float(t["target_mz"])

        if lipid and ion:
            names.append(f"{lipid} {ion} {mz:.4f}")
        elif lipid:
            names.append(f"{lipid} {mz:.4f}")
        else:
            names.append(f"m/z {mz:.4f}")

    return names


# ---------------------------------------------------------------------------
# TIFF output
# ---------------------------------------------------------------------------

def save_imagej_tiff(
    output_tiff: Path,
    stack_to_save: np.ndarray,
    channel_names: Sequence[str],
    source_name: str,
    ppm: float,
) -> None:
    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    tiff.imwrite(
        str(output_tiff),
        stack_to_save,
        imagej=True,
        metadata={
            "axes": "CYX",
            "Channel": list(channel_names),
            "Labels": list(channel_names),
            "Info": f"Extracted from {source_name} with ±{ppm} ppm window",
        },
    )


# ---------------------------------------------------------------------------
# File-pair helpers
# ---------------------------------------------------------------------------

def ensure_imzml_ibd_pair_names(
    imzml_path: Path,
    ibd_path: Path,
) -> Tuple[Path, Path]:
    """
    pyimzML expects the .ibd file next to the .imzML with the same stem.
    Copy the IBD to match the imzML stem when needed.
    """
    expected_ibd = imzml_path.with_suffix(".ibd")

    if ibd_path.resolve() != expected_ibd.resolve():
        shutil.copyfile(ibd_path, expected_ibd)
        return imzml_path, expected_ibd

    return imzml_path, ibd_path


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_multichannel_imzml_to_stack(
    imzml_path: Path,
    targets: List[Dict[str, object]],
    ppm: float,
    dtype_name: str,
    normalize: Optional[str],
    progress_bar=None,
    status_box=None,
) -> Tuple[np.ndarray, List[str]]:
    if status_box is not None:
        status_box.info("Opening imzML file...")

    parser = ImzMLParser(str(imzml_path))
    coords, x_to_idx, y_to_idx, H, W = build_coordinate_maps(parser)

    C = len(targets)

    if status_box is not None:
        status_box.info(
            f"Extracting {C:,} channel(s) from {len(coords):,} spectra. "
            f"Output size: {C} × {H} × {W}"
        )

    stack = np.zeros((C, H, W), dtype=np.float32)
    total = len(coords)

    for spectrum_idx, (x, y, _z) in enumerate(coords):
        mzs, intensities = parser.getspectrum(spectrum_idx)

        xi = x_to_idx[x]
        yi = y_to_idx[y]

        for c, target in enumerate(targets):
            stack[c, yi, xi] = extract_peak_intensity(
                mzs,
                intensities,
                float(target["target_mz"]),
                ppm=ppm,
            )

        if progress_bar is not None and (
            spectrum_idx == 0
            or (spectrum_idx + 1) % 50 == 0
            or spectrum_idx + 1 == total
        ):
            progress_bar.progress(
                (spectrum_idx + 1) / total,
                text=f"Extracting spectra: {spectrum_idx + 1:,}/{total:,}",
            )

    stack = normalize_stack(stack, normalize)
    stack = convert_stack_dtype(stack, dtype_name, normalize)

    return stack, channel_names_from_targets(targets)
