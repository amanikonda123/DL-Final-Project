#!/usr/bin/env python3
"""Generate all report-ready visualization figures locally.

Usage:  python generate_figures.py
"""
import os, sys, yaml, glob
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm, colors as mcolors
from sklearn.manifold import TSNE
from collections import defaultdict

# ── Global font sizes for report readability ─────────────────────────────────
plt.rcParams.update({
    "font.size":          16,
    "axes.titlesize":     18,
    "axes.labelsize":     16,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "legend.fontsize":    14,
    "figure.titlesize":   20,
    "figure.dpi":         150,
})

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from data.loader import get_dataloaders
from data.features import select_features, get_feature_dims
from models import build_model

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")  # SchNet radius_graph requires CPU
DATA_ROOT  = os.path.join(PROJECT_DIR, "data", "qm9_raw")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots", "analysis")
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Plots → {PLOT_DIR}")

# ── Load configs ─────────────────────────────────────────────────────────────
with open("config/gcn.yaml")    as f: gcn_cfg    = yaml.safe_load(f)
with open("config/gat.yaml")    as f: gat_cfg    = yaml.safe_load(f)
with open("config/gatv2.yaml")  as f: gatv2_cfg  = yaml.safe_load(f)
with open("config/schnet.yaml") as f: schnet_cfg = yaml.safe_load(f)

# ── Load data (topo for GCN, full for GAT/GATv2, geo for SchNet) ────────────
topo_train, topo_val, topo_test, topo_norm = get_dataloaders(gcn_cfg, root=DATA_ROOT)
full_train, full_val, full_test, full_norm = get_dataloaders(gat_cfg, root=DATA_ROOT)
geo_train,  geo_val,  geo_test,  geo_norm  = get_dataloaders(schnet_cfg, root=DATA_ROOT)

# ── Build and load models ────────────────────────────────────────────────────
def load_model(name, cfg, ckpt_path):
    fd = get_feature_dims(cfg["dataset"].get("feature_mode", "topology"))
    model = build_model(name, cfg, fd).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    pc = sum(p.numel() for p in model.parameters())
    print(f"  {name.upper():8s} loaded  ({pc:>9,} params)  ← {os.path.basename(ckpt_path)}")
    return model

print("\nLoading checkpoints …")
gcn_model    = load_model("gcn",    gcn_cfg,    f"{OUTPUT_DIR}/checkpoints/gcn_best.pt")
gat_model    = load_model("gat",    gat_cfg,    f"{OUTPUT_DIR}/checkpoints/gat_final_best.pt")
gatv2_model  = load_model("gatv2",  gatv2_cfg,  f"{OUTPUT_DIR}/checkpoints/gatv2_best.pt")

# SchNet needs CPU because radius_graph (torch-cluster) has no MPS kernel
def load_model_cpu(name, cfg, ckpt_path):
    fd = get_feature_dims(cfg["dataset"].get("feature_mode", "topology"))
    model = build_model(name, cfg, fd).to(DEVICE_CPU)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE_CPU, weights_only=True))
    model.eval()
    pc = sum(p.numel() for p in model.parameters())
    print(f"  {name.upper():8s} loaded  ({pc:>9,} params)  ← {os.path.basename(ckpt_path)}  [CPU]")
    return model

schnet_model = load_model_cpu("schnet", schnet_cfg, f"{OUTPUT_DIR}/checkpoints/schnet_best.pt")
print("All models loaded.\n")

# helper: pick correct loader / norm / feature_mode per model
def model_ctx(name):
    if name == "gcn":
        return topo_test, topo_norm, "topology"
    if name in ("gat", "gatv2"):
        return full_test, full_norm, "full"
    return geo_test, geo_norm, "geometry"

MODELS = [
    ("gcn",    gcn_model),
    ("gat",    gat_model),
    ("gatv2",  gatv2_model),
    ("schnet", schnet_model),
]
TITLES = {"gcn": "GCN", "gat": "GAT", "gatv2": "GATv2", "schnet": "SchNet"}

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: t-SNE of Learned Graph Embeddings
# ═══════════════════════════════════════════════════════════════════════════════
N_SAMPLES = 5000

def _dev(model_name):
    return DEVICE_CPU if model_name == "schnet" else DEVICE

def collect_embeddings(model, loader, normalizer, model_name, feature_mode, n=N_SAMPLES):
    embeddings, targets = [], []
    count = 0
    dev = _dev(model_name)
    with torch.no_grad():
        for batch in loader:
            batch = select_features(batch, feature_mode)
            batch = batch.to(dev)
            if model_name == "schnet":
                emb, _ = model.get_embedding(batch.z, batch.pos, batch.batch)
            elif model_name == "gatv2":
                emb, _ = model.get_embedding(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                emb, _ = model.get_embedding(batch.x, batch.edge_index, batch.batch)
            embeddings.append(emb.cpu())
            targets.append(normalizer.denormalize(batch.y.cpu()))
            count += batch.num_graphs
            if count >= n:
                break
    return torch.cat(embeddings)[:n].numpy(), torch.cat(targets)[:n].numpy()

print("── Figure 1: t-SNE ──")
tsne_data = {}
for name, model in MODELS:
    loader, norm, fm = model_ctx(name)
    emb, tgt = collect_embeddings(model, loader, norm, name, fm)
    tsne_data[name] = (
        TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb),
        tgt,
    )
    print(f"  {name} done")

fig = plt.figure(figsize=(24, 6))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.10)
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
cax  = fig.add_subplot(gs[0, 4])
fig.suptitle("t-SNE of Learned Graph Embeddings — Colored by True $U_0$ (Ha)", y=1.02)

for ax, name in zip(axes, ["gcn", "gat", "gatv2", "schnet"]):
    tsne, tgt = tsne_data[name]
    sc = ax.scatter(tsne[:, 0], tsne[:, 1], c=tgt.ravel(), cmap="viridis",
                    s=5, alpha=0.6, rasterized=True)
    ax.set_title(TITLES[name])
    ax.set_xticks([]); ax.set_yticks([])

fig.colorbar(sc, cax=cax, label="True $U_0$ (Ha)")
plt.savefig(f"{PLOT_DIR}/tsne_embeddings.png", bbox_inches="tight")
plt.close()
print("  Saved tsne_embeddings.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Attention Weights — GAT vs GATv2
# ═══════════════════════════════════════════════════════════════════════════════
import networkx as nx
from torch_geometric.utils import subgraph

ATOM_LABELS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

test_mols = []
for batch in full_test:
    for i in range(batch.num_graphs):
        mask = batch.batch == i
        node_idx = mask.nonzero(as_tuple=True)[0]
        mol = batch.__class__()
        mol.x = batch.x[mask]
        mol.z = batch.z[mask]
        ei, _, emask = subgraph(mask, batch.edge_index, relabel_nodes=True, return_edge_mask=True)
        mol.edge_index = ei
        mol.edge_attr = batch.edge_attr[emask] if batch.edge_attr is not None else None
        mol.y = batch.y[i:i+1]
        mol.batch = torch.zeros(mask.sum().item(), dtype=torch.long)
        test_mols.append(mol)
        if len(test_mols) >= 200:
            break
    if len(test_mols) >= 200:
        break

sizes = [m.x.size(0) for m in test_mols]
selected = [
    test_mols[min(range(len(sizes)), key=lambda i: abs(sizes[i] - t))]
    for t in [5, 12, 18, 25]
]

def draw_attention(ax, mol, ei, alpha_mean, title_prefix):
    G = nx.Graph()
    n_nodes = mol.x.size(0)
    for n in range(n_nodes):
        G.add_node(n)
    ei_cpu = ei.cpu().numpy()
    edge_weights = {}
    for e in range(ei_cpu.shape[1]):
        src, dst = int(ei_cpu[0, e]), int(ei_cpu[1, e])
        if src < dst:
            key = (src, dst)
            edge_weights[key] = edge_weights.get(key, 0) + alpha_mean[e]
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    labels = {n: ATOM_LABELS.get(mol.z[n].item(), "?") for n in range(n_nodes)}
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    widths = [3.0 * w / max_w + 0.5 for w in weights]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=350, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.7, edge_color="steelblue")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=11)
    ax.set_title(f"{title_prefix} — {n_nodes} atoms")
    ax.axis("off")

print("── Figure 2: Attention ──")
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle("Attention Weights — Last Layer (edge thickness ∝ attention)\n"
             "Top: GAT (node-only)  |  Bottom: GATv2 (edge-aware)", y=1.03)

for col, mol in enumerate(selected):
    mol_feat = select_features(mol.to(DEVICE), "full")
    gat_attn = gat_model.get_attention_weights(mol_feat.x, mol_feat.edge_index, mol_feat.batch)
    ei_g, al_g = gat_attn[-1]
    draw_attention(axes[0, col], mol, ei_g, al_g.mean(dim=-1).cpu().numpy(), "GAT")
    gatv2_attn = gatv2_model.get_attention_weights(
        mol_feat.x, mol_feat.edge_index, mol_feat.edge_attr, mol_feat.batch)
    ei_v, al_v = gatv2_attn[-1]
    draw_attention(axes[1, col], mol, ei_v, al_v.mean(dim=-1).cpu().numpy(), "GATv2")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/gat_attention.png", bbox_inches="tight")
plt.close()
print("  Saved gat_attention.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Over-smoothing Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════
from torch.nn.functional import cosine_similarity

def compute_layer_similarity(model, loader, model_name, feature_mode, n_batches=10):
    layer_sims = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = select_features(batch, feature_mode)
            batch = batch.to(DEVICE)
            if model_name == "schnet":
                break
            elif model_name == "gatv2":
                _, le = model.get_embedding(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                _, le = model.get_embedding(batch.x, batch.edge_index, batch.batch)
            if layer_sims is None:
                layer_sims = [0.0] * len(le)
            for li, emb in enumerate(le):
                en = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                sm = en @ en.T
                n = en.size(0)
                mask = ~torch.eye(n, dtype=torch.bool, device=sm.device)
                layer_sims[li] += sm[mask].mean().item()
            count += 1
            if count >= n_batches:
                break
    return [s / count for s in layer_sims] if layer_sims else []

print("── Figure 3: Over-smoothing ──")
gcn_sims   = compute_layer_similarity(gcn_model,   topo_test, "gcn",   "topology")
gat_sims   = compute_layer_similarity(gat_model,   full_test, "gat",   "full")
gatv2_sims = compute_layer_similarity(gatv2_model, full_test, "gatv2", "full")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(range(len(gcn_sims)),   gcn_sims,   "o-", label="GCN",   linewidth=2.5, markersize=9)
ax.plot(range(len(gat_sims)),   gat_sims,   "s-", label="GAT",   linewidth=2.5, markersize=9)
ax.plot(range(len(gatv2_sims)), gatv2_sims, "D-", label="GATv2", linewidth=2.5, markersize=9)
ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Perfect similarity")
ax.set_xlabel("Layer Index")
ax.set_ylabel("Mean Pairwise Cosine Similarity")
ax.set_title("Over-smoothing Diagnostic — Node Embedding Similarity per Layer")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/oversmoothing.png", bbox_inches="tight")
plt.close()
print("  Saved oversmoothing.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Error vs Molecule Size
# ═══════════════════════════════════════════════════════════════════════════════
print("── Figure 4: Error vs Size ──")

def collect_errors_by_size(model, loader, normalizer, model_name, feature_mode):
    sizes_list, errors_list = [], []
    dev = _dev(model_name)
    with torch.no_grad():
        for batch in loader:
            bp = select_features(batch, feature_mode).to(dev)
            if model_name == "schnet":
                pred = model(bp.z, bp.pos, bp.batch)
            elif model_name == "gatv2":
                pred = model(bp.x, bp.edge_index, bp.edge_attr, bp.batch)
            else:
                pred = model(bp.x, bp.edge_index, bp.batch)
            pd_ = normalizer.denormalize(pred.cpu())
            td_ = normalizer.denormalize(bp.y.cpu())
            for i in range(batch.num_graphs):
                n_a = (batch.batch == i).sum().item()
                sizes_list.append(n_a)
                errors_list.append(abs(pd_[i].item() - td_[i].item()))
    return np.array(sizes_list), np.array(errors_list)

err_data = {}
for name, model in MODELS:
    loader, norm, fm = model_ctx(name)
    err_data[name] = collect_errors_by_size(model, loader, norm, name, fm)
    print(f"  {name} done")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle("Prediction Error vs Molecule Size (Number of Atoms)", y=1.02)

for ax, name in zip(axes, ["gcn", "gat", "gatv2", "schnet"]):
    sz, er = err_data[name]
    ax.scatter(sz, er, s=4, alpha=0.3, rasterized=True)
    usizes = sorted(set(sz))
    bmeans = [np.mean(er[sz == s]) for s in usizes]
    ax.plot(usizes, bmeans, "r-", linewidth=2.5, label="Mean error")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Absolute Error (Ha)")
    ax.set_title(TITLES[name])
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_vs_size.png", bbox_inches="tight")
plt.close()
print("  Saved error_vs_size.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Predicted vs True
# ═══════════════════════════════════════════════════════════════════════════════
print("── Figure 5: Predicted vs True ──")

def collect_predictions(model, loader, normalizer, model_name, feature_mode):
    preds, tgts = [], []
    dev = _dev(model_name)
    with torch.no_grad():
        for batch in loader:
            batch = select_features(batch, feature_mode).to(dev)
            if model_name == "schnet":
                pred = model(batch.z, batch.pos, batch.batch)
            elif model_name == "gatv2":
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(normalizer.denormalize(pred.cpu()))
            tgts.append(normalizer.denormalize(batch.y.cpu()))
    return torch.cat(preds).numpy(), torch.cat(tgts).numpy()

pred_data = {}
for name, model in MODELS:
    loader, norm, fm = model_ctx(name)
    pred_data[name] = collect_predictions(model, loader, norm, name, fm)
    print(f"  {name} done")

all_errors = np.concatenate([
    np.abs(pred_data[n][0].ravel() - pred_data[n][1].ravel()) for n in pred_data
])
vmax_clip = np.percentile(all_errors, 95)

all_true = np.concatenate([pred_data[n][1].ravel() for n in pred_data])
shared_lo = float(all_true.min()) * 1.05   # 5 % padding (values are negative)
shared_hi = float(all_true.max()) * 0.95

fig = plt.figure(figsize=(24, 6))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.25)
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
cax  = fig.add_subplot(gs[0, 4])
fig.suptitle("Predicted vs True $U_0$ — Test Set", y=1.02)

for i, (ax, name) in enumerate(zip(axes, ["gcn", "gat", "gatv2", "schnet"])):
    pred, true = pred_data[name]
    err = np.abs(pred.ravel() - true.ravel())
    sc = ax.scatter(true.ravel(), pred.ravel(), c=err, cmap="coolwarm",
                    vmin=0, vmax=vmax_clip, s=4, alpha=0.5, rasterized=True)
    ax.plot([shared_lo, shared_hi], [shared_lo, shared_hi],
            "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(shared_lo, shared_hi)
    ax.set_ylim(shared_lo, shared_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("True $U_0$ (Ha)")
    if i == 0:
        ax.set_ylabel("Predicted $U_0$ (Ha)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_title(f"{TITLES[name]}  |  MAE={np.mean(err):.2f}")
    ax.grid(alpha=0.3)

fig.colorbar(sc, cax=cax, label="Absolute Error (Ha)")
plt.savefig(f"{PLOT_DIR}/predicted_vs_true.png", bbox_inches="tight")
plt.close()
print("  Saved predicted_vs_true.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: SchNet 3D Interaction Visualization
# ═══════════════════════════════════════════════════════════════════════════════
print("── Figure 6: SchNet 3D ──")

geo_mols = []
for batch in geo_test:
    for i in range(batch.num_graphs):
        mask = batch.batch == i
        mol = batch.__class__()
        mol.z = batch.z[mask]; mol.pos = batch.pos[mask]; mol.y = batch.y[i:i+1]
        mol.batch = torch.zeros(mask.sum().item(), dtype=torch.long)
        ei, _ = subgraph(mask, batch.edge_index, relabel_nodes=True)
        mol.edge_index = ei
        geo_mols.append(mol)
        if len(geo_mols) >= 200: break
    if len(geo_mols) >= 200: break

gsizes = [m.z.size(0) for m in geo_mols]
selected_geo = [
    geo_mols[min(range(len(gsizes)), key=lambda i: abs(gsizes[i] - t))]
    for t in [5, 12, 18, 25]
]

def summarize_interactions(edge_index, edge_weight, edge_strength):
    pair_scores, pair_dists = defaultdict(list), defaultdict(list)
    ei = edge_index.cpu().numpy()
    ew = edge_weight.cpu().numpy()
    es = edge_strength.cpu().numpy()
    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        if src == dst: continue
        key = tuple(sorted((src, dst)))
        pair_scores[key].append(float(es[k]))
        pair_dists[key].append(float(ew[k]))
    summary = [(k, float(np.mean(v)), float(np.mean(pair_dists[k])))
               for k, v in pair_scores.items()]
    summary.sort(key=lambda x: x[1], reverse=True)
    return summary

def unique_bonds(edge_index):
    bonds = set()
    ei = edge_index.cpu().numpy()
    for k in range(ei.shape[1]):
        s, d = int(ei[0, k]), int(ei[1, k])
        if s != d: bonds.add(tuple(sorted((s, d))))
    return sorted(bonds)

def set_equal_3d_axes(ax, pos):
    mins, maxs = pos.min(axis=0), pos.max(axis=0)
    c = (mins + maxs) / 2.0
    r = max((maxs - mins).max() / 2.0, 1e-3)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1))

def draw_schnet_3d(ax, mol, title):
    mol_dev = mol.to(DEVICE_CPU)
    info = schnet_model.get_interaction_graph(mol_dev.z, mol_dev.pos, mol_dev.batch)
    pos = info["pos"].cpu().numpy()
    astr = info["atom_strength"].cpu().numpy()
    interactions = summarize_interactions(info["edge_index"], info["edge_weight"], info["edge_strength"])
    for u, v in unique_bonds(mol.edge_index):
        xyz = pos[[u, v]]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="lightgray", linewidth=1.2, alpha=0.7, zorder=1)
    top_k = min(max(5, mol.z.size(0) // 2), 10, len(interactions))
    top_int = interactions[:top_k]
    if top_int:
        strengths = np.array([s for _, s, _ in top_int])
        ne = mcolors.Normalize(vmin=strengths.min(), vmax=strengths.max() + 1e-8)
        ecmap = cm.plasma
        for (u, v), strength, _ in top_int:
            xyz = pos[[u, v]]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ecmap(ne(strength)),
                    linewidth=1.5 + 3.5 * ne(strength), alpha=0.95, zorder=2)
    anorm = mcolors.Normalize(vmin=astr.min(), vmax=astr.max() + 1e-8)
    asizes = 90 + 180 * anorm(astr)
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=astr, cmap="viridis",
                         s=asizes, edgecolors="black", linewidths=0.6, alpha=0.98, zorder=3)
    for idx, (x, y, z) in enumerate(pos):
        anum = int(mol.z[idx])
        if anum != 1:
            ax.text(x, y, z, ATOM_LABELS.get(anum, "?"), fontsize=11, zorder=4)
    set_equal_3d_axes(ax, pos)
    ax.set_title(f"{title} — {mol.z.size(0)} atoms", pad=12)
    ax.view_init(elev=22, azim=35)
    ax.set_axis_off()
    return scatter

fig = plt.figure(figsize=(18, 14))
axes3d = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
fig.suptitle("SchNet 3D Geometric Interaction View", y=0.97)
last_sc = None
for ax, mol, title in zip(axes3d, selected_geo, ["Small", "Medium", "Large", "XL"]):
    last_sc = draw_schnet_3d(ax, mol, title)
fig.subplots_adjust(left=0.03, right=0.9, top=0.92, bottom=0.04, wspace=0.08, hspace=0.18)
cax3d = fig.add_axes([0.92, 0.2, 0.018, 0.6])
cbar = fig.colorbar(last_sc, cax=cax3d)
cbar.set_label("SchNet atom-state magnitude")
plt.savefig(f"{PLOT_DIR}/schnet_3d_interactions.png", dpi=160)
plt.close()
print("  Saved schnet_3d_interactions.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: Training Curves
# ═══════════════════════════════════════════════════════════════════════════════
print("── Figure 7: Training Curves ──")

HISTORY_MAP = {
    "gcn":    "gcn_history.csv",
    "gat":    "gat_final_history.csv",
    "gatv2":  "gatv2_trial_004_history.csv",
    "schnet": "schnet_history.csv",
}
COLORS_MAP = {"gcn": "#4C72B0", "gat": "#DD8452", "gatv2": "#55A868", "schnet": "#C44E52"}

histories = {}
for name, fname in HISTORY_MAP.items():
    path = os.path.join(OUTPUT_DIR, "logs", fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 0:
            histories[name] = df
            print(f"  {name}: {len(df)} epochs")

if histories:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Training Curves — All Models")

    for name, df in histories.items():
        c = COLORS_MAP.get(name, "grey")
        axes[0].plot(df["epoch"], df["val_loss"],   label=f"{name.upper()} val",
                     color=c, linewidth=2.5)
        axes[0].plot(df["epoch"], df["train_loss"],  label=f"{name.upper()} train",
                     color=c, linewidth=1.5, linestyle="--", alpha=0.5)
        axes[1].plot(df["epoch"], df["val_mae"], label=name.upper(),
                     color=c, linewidth=2.5)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss (normalized)")
    axes[0].set_title("Train / Val Loss"); axes[0].legend(ncol=2); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val MAE (Ha)")
    axes[1].set_title("Validation MAE"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/training_curves_all.png", bbox_inches="tight")
    plt.close()
    print("  Saved training_curves_all.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ALL FIGURES GENERATED")
print("=" * 60)
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
