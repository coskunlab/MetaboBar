"""
GNN explainability utility module.
Supports binary (positivity) and multiclass (clustering) node-level
classification on mixed cell+superpixel spatial graphs.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv
from torch_geometric.utils import coalesce

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))


def build_radius_edges(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    radius: float,
    idx_a_offset: int = 0,
    idx_b_offset: int = 0,
    self_mode: bool = False,
) -> np.ndarray:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.empty((2, 0), dtype=np.int64)
    tree_b = cKDTree(coords_b)
    pairs = tree_b.query_ball_point(coords_a, r=radius)
    src_list, dst_list = [], []
    for i, neighbours in enumerate(pairs):
        for j in neighbours:
            if self_mode and i == j and idx_a_offset == idx_b_offset:
                continue
            src_list.append(i + idx_a_offset)
            dst_list.append(j + idx_b_offset)
    if not src_list:
        return np.empty((2, 0), dtype=np.int64)
    return np.array([src_list, dst_list], dtype=np.int64)


def make_bidirectional(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return edge_index
    return np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)


def coalesce_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return coalesce(edge_index, num_nodes=num_nodes)


def compute_metrics_binary(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def compute_metrics_multiclass(y_true, y_pred):
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def summarize_metric(values):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem  = std / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    return mean, std, sem


def sample_indices(idx, max_n, seed):
    if max_n is None or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_n, replace=False)


# ---------------------------------------------------------------------------
# GNN Models
# ---------------------------------------------------------------------------

def _make_conv(model_type, in_dim, out_dim, heads=4):
    if model_type == "GraphSAGE":
        return SAGEConv(in_dim, out_dim)
    if model_type == "GCN":
        return GCNConv(in_dim, out_dim)
    if model_type == "GATv2":
        return GATv2Conv(in_dim, out_dim, heads=heads, concat=True)
    raise ValueError(f"Unknown model_type: {model_type!r}")


class GNNBinaryClassifier(nn.Module):
    _GAT_HEADS = 4

    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.2, model_type="GraphSAGE"):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        gat_heads = self._GAT_HEADS
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else (gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim)
            self.convs.append(_make_conv(model_type, layer_in, hidden_dim, heads=gat_heads))
            bn_in = gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim
            self.bns.append(nn.BatchNorm1d(bn_in))
        final_dim = gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim
        self.head = nn.Linear(final_dim, 1)

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        return self.head(self.encode(x, edge_index)).squeeze(-1)


class GNNMulticlassClassifier(nn.Module):
    _GAT_HEADS = 4

    def __init__(self, in_dim, num_classes, hidden_dim=64, num_layers=3, dropout=0.2, model_type="GraphSAGE"):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        gat_heads = self._GAT_HEADS
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else (gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim)
            self.convs.append(_make_conv(model_type, layer_in, hidden_dim, heads=gat_heads))
            bn_in = gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim
            self.bns.append(nn.BatchNorm1d(bn_in))
        final_dim = gat_heads * hidden_dim if model_type == "GATv2" else hidden_dim
        self.head = nn.Linear(final_dim, num_classes)

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        return self.head(self.encode(x, edge_index))


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def explain_saliency(model, data, node_indices, class_idx=None):
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    logits = model(x, data.edge_index)
    score = logits[node_indices].mean() if class_idx is None else logits[node_indices, class_idx].mean()
    score.backward()
    return x.grad[node_indices].abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)


def explain_occlusion(model, data, node_indices, class_idx=None):
    model.eval()
    feat_dim = data.x.shape[1]
    with torch.no_grad():
        base = model(data.x, data.edge_index)
        base_score = base[node_indices].mean().item() if class_idx is None else base[node_indices, class_idx].mean().item()
    importance = np.zeros(feat_dim, dtype=np.float32)
    for f in range(feat_dim):
        x_occ = data.x.clone()
        x_occ[:, f] = 0.0
        with torch.no_grad():
            occ = model(x_occ, data.edge_index)
            occ_score = occ[node_indices].mean().item() if class_idx is None else occ[node_indices, class_idx].mean().item()
        importance[f] = base_score - occ_score
    return importance


def explain_nodes(model, data, node_indices, method, class_idx=None):
    if method == "saliency":
        return explain_saliency(model, data, node_indices, class_idx)
    if method == "occlusion":
        return explain_occlusion(model, data, node_indices, class_idx)
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def _find_coord_cols(df):
    for cx, cy in [("centroid_x", "centroid_y"), ("x", "y"), ("cx", "cy"), ("X", "Y")]:
        if cx in df.columns and cy in df.columns:
            return cx, cy
    raise ValueError(f"Cannot find coordinate columns in {list(df.columns)}")


def build_mixed_graph(cell_df, sp_df, cell_id_col, sp_id_col, cx_col, cy_col,
                      radius_um, pixel_size_um,
                      include_cell_cell=True, include_cell_sp=True,
                      include_sp_sp=True, bidirectional=True):
    cell_xy = cell_df[[cx_col, cy_col]].values.astype(np.float64) * pixel_size_um
    sp_xy   = sp_df[[cx_col, cy_col]].values.astype(np.float64) * pixel_size_um
    n_cells, n_sp = len(cell_xy), len(sp_xy)
    n_nodes = n_cells + n_sp

    parts = []
    if include_cell_cell and n_cells > 0:
        parts.append(build_radius_edges(cell_xy, cell_xy, radius_um, 0, 0, self_mode=True))
    if include_cell_sp and n_cells > 0 and n_sp > 0:
        parts.append(build_radius_edges(cell_xy, sp_xy, radius_um, 0, n_cells))
    if include_sp_sp and n_sp > 0:
        parts.append(build_radius_edges(sp_xy, sp_xy, radius_um, n_cells, n_cells, self_mode=True))

    edge_np = np.concatenate(parts, axis=1) if parts else np.empty((2, 0), dtype=np.int64)
    if bidirectional:
        edge_np = make_bidirectional(edge_np)

    edge_tensor = coalesce_edges(torch.from_numpy(edge_np).long(), n_nodes)
    node_type = np.array([0] * n_cells + [1] * n_sp, dtype=np.int32)
    node_ids  = np.concatenate([cell_df[cell_id_col].values, sp_df[sp_id_col].values])
    coords_all = np.vstack([cell_xy, sp_xy])
    return edge_tensor, cell_xy, sp_xy, n_cells, n_sp, n_nodes, node_type, node_ids, coords_all


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(cell_df, sp_df, feature_cols, standardize=True, log1p=False):
    X = np.vstack([
        cell_df[feature_cols].values.astype(np.float32),
        sp_df[feature_cols].values.astype(np.float32),
    ])
    if log1p:
        X = np.log1p(np.clip(X, 0, None))
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    return X, scaler


# ---------------------------------------------------------------------------
# K-fold splits
# ---------------------------------------------------------------------------

def make_kfold_splits(y_cells, n_folds, inner_val_frac=0.15, seed=42):
    n = len(y_cells)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(n), y_cells)):
        rng = np.random.default_rng(seed + fold)
        n_val = max(1, int(len(trainval_idx) * inner_val_frac))
        val_local = rng.choice(len(trainval_idx), size=n_val, replace=False)
        train_local = np.setdiff1d(np.arange(len(trainval_idx)), val_local)
        train_idx = trainval_idx[train_local]
        val_idx   = trainval_idx[val_local]
        tm = np.zeros(n, dtype=bool); tm[train_idx] = True
        vm = np.zeros(n, dtype=bool); vm[val_idx]   = True
        em = np.zeros(n, dtype=bool); em[test_idx]  = True
        splits.append((fold, tm, vm, em))
    return splits


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _plot_importance(imp_df, top_k, title, save_path):
    df = imp_df.sort_values("mean_importance", ascending=False).head(top_k).iloc[::-1]
    sem = df.get("sem", pd.Series(np.zeros(len(df)), index=df.index))
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(df["feature"], df["mean_importance"], xerr=sem,
            color="steelblue", ecolor="black", capsize=3, height=0.6)
    ax.set_xlabel("Mean importance")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Binary fold training
# ---------------------------------------------------------------------------

def train_one_fold_binary(base_data, y_all, feature_names, target_name, fold, fold_dir,
                          n_cells, n_nodes, feat_dim, node_type, node_ids,
                          train_mask_cells, val_mask_cells, test_mask_cells,
                          hidden_dim, num_layers, dropout, lr, weight_decay,
                          epochs, patience, explain_method, explain_on,
                          max_explain_nodes, top_k, device, seed,
                          save_model=True, status_cb=None):
    set_seed(seed + fold)
    dev = torch.device(device)

    train_mask = np.zeros(n_nodes, dtype=bool); train_mask[:n_cells] = train_mask_cells
    val_mask   = np.zeros(n_nodes, dtype=bool); val_mask[:n_cells]   = val_mask_cells
    test_mask  = np.zeros(n_nodes, dtype=bool); test_mask[:n_cells]  = test_mask_cells

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    y_train = y_all[train_idx]
    n_pos = float(np.sum(y_train == 1)); n_neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32).to(dev)

    model = GNNBinaryClassifier(feat_dim, hidden_dim, num_layers, dropout).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x = base_data.x.to(dev)
    ei = base_data.edge_index.to(dev)
    y_t = torch.tensor(y_all, dtype=torch.float32).to(dev)
    train_t = torch.tensor(train_idx, dtype=torch.long).to(dev)
    val_t   = torch.tensor(val_idx,   dtype=torch.long).to(dev)

    best_val_auc, best_state, patience_ctr = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(x, ei)[train_t], y_t[train_t])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x, ei)[val_t]).cpu().numpy()
        try:
            val_auc = roc_auc_score(y_all[val_idx].astype(int), val_prob)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            if status_cb: status_cb(f"  Fold {fold}: early stop epoch {epoch} (val_auc={best_val_auc:.4f})")
            break
        if epoch % 25 == 0 and status_cb:
            status_cb(f"  Fold {fold} ep {epoch}/{epochs} loss={loss.item():.4f} val_auc={val_auc:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    # No model weights, metrics, or predictions saved — importance only
    model.eval()
    with torch.no_grad():
        all_logits = model(x, ei)

    if explain_on == "test_positives":
        exp_idx = np.where(test_mask_cells & (y_all[:n_cells] == 1))[0]
    else:
        exp_idx = np.where(test_mask_cells)[0]
    if len(exp_idx) == 0:
        exp_idx = np.where(test_mask_cells)[0]
    exp_idx = sample_indices(exp_idx, max_explain_nodes, seed + fold)

    exp_data = Data(x=base_data.x.to(dev), edge_index=base_data.edge_index.to(dev))
    importance = explain_nodes(model, exp_data, exp_idx, explain_method) if len(exp_idx) > 0 else np.zeros(feat_dim, np.float32)

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance, "fold": fold})
    return imp_df


# ---------------------------------------------------------------------------
# Multiclass fold training
# ---------------------------------------------------------------------------

def train_one_fold_multiclass(base_data, y_all, feature_names, target_name, fold, fold_dir,
                              n_cells, n_nodes, feat_dim, node_type, node_ids,
                              train_mask_cells, val_mask_cells, test_mask_cells,
                              num_classes, class_names,
                              hidden_dim, num_layers, dropout, lr, weight_decay,
                              epochs, patience, explain_method, max_explain_nodes,
                              top_k, device, seed, save_model=True, status_cb=None):
    set_seed(seed + fold)
    dev = torch.device(device)

    train_mask = np.zeros(n_nodes, dtype=bool); train_mask[:n_cells] = train_mask_cells
    val_mask   = np.zeros(n_nodes, dtype=bool); val_mask[:n_cells]   = val_mask_cells
    test_mask  = np.zeros(n_nodes, dtype=bool); test_mask[:n_cells]  = test_mask_cells

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    y_train = y_all[train_idx].astype(int)
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    cw = torch.tensor(1.0 / counts / (1.0 / counts).sum() * num_classes, dtype=torch.float32).to(dev)

    model = GNNMulticlassClassifier(feat_dim, num_classes, hidden_dim, num_layers, dropout).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cw)

    x = base_data.x.to(dev)
    ei = base_data.edge_index.to(dev)
    y_t = torch.tensor(y_all, dtype=torch.long).to(dev)
    train_t = torch.tensor(train_idx, dtype=torch.long).to(dev)
    val_t   = torch.tensor(val_idx,   dtype=torch.long).to(dev)

    best_val_f1, best_state, patience_ctr = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(x, ei)[train_t], y_t[train_t])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x, ei)[val_t].argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_all[val_idx].astype(int), val_pred, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= patience:
            if status_cb: status_cb(f"  Fold {fold}: early stop epoch {epoch} (val_f1={best_val_f1:.4f})")
            break
        if epoch % 25 == 0 and status_cb:
            status_cb(f"  Fold {fold} ep {epoch}/{epochs} loss={loss.item():.4f} val_f1={val_f1:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    # No model weights, metrics, or predictions saved — importance only
    model.eval()
    with torch.no_grad():
        all_logits = model(x, ei)

    exp_data = Data(x=base_data.x.to(dev), edge_index=base_data.edge_index.to(dev))
    imp_per_class: Dict[str, pd.DataFrame] = {}
    for c, cn in enumerate(class_names):
        exp_idx = np.where(test_mask_cells & (y_all[:n_cells] == c))[0]
        if len(exp_idx) == 0:
            exp_idx = np.where(test_mask_cells)[0]
        exp_idx = sample_indices(exp_idx, max_explain_nodes, seed + fold + c)
        importance = explain_nodes(model, exp_data, exp_idx, explain_method, class_idx=c) if len(exp_idx) > 0 else np.zeros(feat_dim, np.float32)
        imp_per_class[cn] = pd.DataFrame({"feature": feature_names, "importance": importance, "fold": fold, "cluster": cn})

    return imp_per_class

# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def _build_base(cell_df, sp_df, feature_cols, radius_um, pixel_size_um,
                standardize, log1p, output_dir, status_cb):
    cell_id_col = cell_df.columns[0]
    sp_id_col   = sp_df.columns[0]
    cx_col, cy_col = _find_coord_cols(cell_df)

    if status_cb: status_cb("Building mixed cell+superpixel graph…")
    (edge_tensor, cell_xy, sp_xy, n_cells, n_sp, n_nodes,
     node_type, node_ids, coords_all) = build_mixed_graph(
        cell_df, sp_df, cell_id_col, sp_id_col, cx_col, cy_col,
        radius_um, pixel_size_um)

    if status_cb: status_cb(f"  {n_nodes} nodes ({n_cells} cells, {n_sp} superpixels), {edge_tensor.shape[1]} edges")
    if status_cb: status_cb("Preparing features…")
    X_all, scaler = prepare_features(cell_df, sp_df, feature_cols, standardize, log1p)
    base_data = Data(x=torch.tensor(X_all, dtype=torch.float32), edge_index=edge_tensor)
    return base_data, n_cells, n_sp, n_nodes, node_type, node_ids, X_all.shape[1]


def run_binary_gnn(cell_df, sp_df, feature_cols, binary_labels_col, target_name,
                   output_dir, n_folds=5, inner_val_frac=0.15,
                   radius_um=50.0, pixel_size_um=2.6,
                   hidden_dim=64, num_layers=3, dropout=0.2,
                   lr=1e-3, weight_decay=1e-4, epochs=150, patience=20,
                   model_type="GraphSAGE", explain_method="saliency",
                   explain_on="test_positives", max_explain_nodes=None,
                   top_k=25, standardize=True, log1p=False, seed=42,
                   status_cb=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    target_dir = output_dir / safe_filename(target_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    def _cb(m):
        if status_cb: status_cb(m)

    _cb(f"[Binary GNN] {target_name} | device={device}")
    base_data, n_cells, n_sp, n_nodes, node_type, node_ids, feat_dim = _build_base(
        cell_df, sp_df, feature_cols, radius_um, pixel_size_um, standardize, log1p, output_dir, status_cb)

    y_cells = cell_df[binary_labels_col].values.astype(np.float32)
    y_all   = np.concatenate([y_cells, np.full(n_sp, -1, dtype=np.float32)])
    splits  = make_kfold_splits(y_cells.astype(int), n_folds, inner_val_frac, seed)

    all_imp = []
    for fold, tm, vm, em in splits:
        _cb(f"Fold {fold+1}/{n_folds}…")
        imp_df = train_one_fold_binary(
            base_data, y_all, feature_cols, target_name, fold, target_dir / f"fold_{fold}",
            n_cells, n_nodes, feat_dim, node_type, node_ids, tm, vm, em,
            hidden_dim, num_layers, dropout, lr, weight_decay, epochs, patience,
            explain_method, explain_on, max_explain_nodes, top_k, device, seed,
            status_cb=status_cb)
        all_imp.append(imp_df)

    imp_all  = pd.concat(all_imp, ignore_index=True)
    imp_mean = imp_all.groupby("feature")["importance"].agg(["mean","std"]).reset_index()
    imp_mean.columns = ["feature", "mean_importance", "std_importance"]
    imp_mean["sem"] = imp_mean["std_importance"] / np.sqrt(len(all_imp))
    imp_mean.to_csv(target_dir / "feature_importance_mean.csv", index=False)
    # Save foldwise for comparative violin plots
    imp_all.to_csv(target_dir / "feature_importance_foldwise.csv", index=False)
    _plot_importance(imp_mean, top_k, f"{target_name} – top-{top_k} (binary)",
                     target_dir / "feature_importance_topk.png")

    _cb(f"[Binary GNN] Done → {target_dir}")
    return {"target": target_name, "task": "binary", "output_dir": str(target_dir)}


def run_multiclass_gnn(cell_df, sp_df, feature_cols, cluster_col, target_name,
                       output_dir, n_folds=5, inner_val_frac=0.15,
                       radius_um=50.0, pixel_size_um=2.6,
                       hidden_dim=64, num_layers=3, dropout=0.2,
                       lr=1e-3, weight_decay=1e-4, epochs=150, patience=20,
                       model_type="GraphSAGE", explain_method="saliency",
                       max_explain_nodes=None, top_k=25,
                       standardize=True, log1p=False, seed=42, status_cb=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    target_dir = output_dir / safe_filename(target_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    def _cb(m):
        if status_cb: status_cb(m)

    _cb(f"[Multiclass GNN] {target_name} | device={device}")
    base_data, n_cells, n_sp, n_nodes, node_type, node_ids, feat_dim = _build_base(
        cell_df, sp_df, feature_cols, radius_um, pixel_size_um, standardize, log1p, output_dir, status_cb)

    raw_labels   = cell_df[cluster_col].values
    unique_labels = sorted(np.unique(raw_labels))
    label_to_int  = {lbl: i for i, lbl in enumerate(unique_labels)}
    class_names   = [str(lbl) for lbl in unique_labels]
    num_classes   = len(class_names)
    y_cells = np.array([label_to_int[lbl] for lbl in raw_labels], dtype=np.int64)
    y_all   = np.concatenate([y_cells, np.zeros(n_sp, dtype=np.int64)])
    _cb(f"  {num_classes} classes: {class_names}")

    splits = make_kfold_splits(y_cells, n_folds, inner_val_frac, seed)

    all_imp_per_class: Dict[str, List] = {cn: [] for cn in class_names}

    for fold, tm, vm, em in splits:
        _cb(f"Fold {fold+1}/{n_folds}…")
        imp_dict = train_one_fold_multiclass(
            base_data, y_all, feature_cols, target_name, fold, target_dir / f"fold_{fold}",
            n_cells, n_nodes, feat_dim, node_type, node_ids, tm, vm, em,
            num_classes, class_names,
            hidden_dim, num_layers, dropout, lr, weight_decay, epochs, patience,
            explain_method, max_explain_nodes, top_k, device, seed,
            status_cb=status_cb)
        for cn in class_names:
            if cn in imp_dict:
                all_imp_per_class[cn].append(imp_dict[cn])

    for cn in class_names:
        if not all_imp_per_class[cn]:
            continue
        cn_safe = safe_filename(cn)
        imp_all  = pd.concat(all_imp_per_class[cn], ignore_index=True)
        imp_mean = imp_all.groupby("feature")["importance"].agg(["mean","std"]).reset_index()
        imp_mean.columns = ["feature", "mean_importance", "std_importance"]
        imp_mean["sem"] = imp_mean["std_importance"] / np.sqrt(len(all_imp_per_class[cn]))
        imp_mean["cluster"] = cn
        imp_mean.to_csv(target_dir / f"feature_importance_mean_cluster_{cn_safe}.csv", index=False)
        # Save foldwise for comparative violin plots
        imp_all.rename(columns={"importance": "importance_fold"}).assign(cluster=cn).to_csv(
            target_dir / f"feature_importance_foldwise_cluster_{cn_safe}.csv", index=False
        )
        _plot_importance(imp_mean, top_k, f"{target_name} cluster '{cn}' top-{top_k}",
                         target_dir / f"feature_importance_topk_cluster_{cn_safe}.png")

    _cb(f"[Multiclass GNN] Done → {target_dir}")
    return {"target": target_name, "task": "multiclass", "num_classes": num_classes, "output_dir": str(target_dir)}
