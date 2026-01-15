#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "huggingface_hub",
#     "safetensors",
#     "torch",
#     "scikit-learn",
# ]
# ///
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "Solar": "upstage/Solar-Open-100B",
    "GLM": "zai-org/GLM-4.5-Air",
    "INTELLECT": "PrimeIntellect/INTELLECT-3",
    "Phi": "microsoft/Phi-3.5-MoE-instruct",
}

COLORS = {
    "Solar": "#8B7FC7",
    "GLM": "#4a4a4a",
    "INTELLECT": "#9a9a9a",
    "Phi": "#2ca02c",
}

MARKERS = {
    "Solar": "o",
    "GLM": "s",
    "INTELLECT": "^",
    "Phi": "D",
}


def load_layernorm_weight(model_id: str, layer_idx: int, norm_type: str = "input"):
    try:
        index_file = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)

        if norm_type == "input":
            key = f"model.layers.{layer_idx}.input_layernorm.weight"
        else:
            key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        if key not in index["weight_map"]:
            return None

        shard_name = index["weight_map"][key]
        shard_file = hf_hub_download(repo_id=model_id, filename=shard_name)

        with safe_open(shard_file, framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key).float().numpy()
    except Exception:
        pass
    return None


def load_all_layers(model_id: str, norm_type: str, max_layer: int = 46):
    weights = []
    for layer_idx in range(max_layer + 1):
        w = load_layernorm_weight(model_id, layer_idx, norm_type)
        if w is not None:
            weights.append(w)
    if weights:
        return np.array(weights)
    return None


all_weights = {}
for name, model_id in MODELS.items():
    w = load_all_layers(model_id, "input")
    if w is not None:
        all_weights[name] = w

solar_weights = all_weights.get("Solar")
glm_weights = all_weights.get("GLM")
intellect_weights = all_weights.get("INTELLECT")
phi_weights = all_weights.get("Phi")

if solar_weights is None or glm_weights is None:
    raise SystemExit("Failed to load Solar or GLM weights.")

n_layers = min(len(solar_weights), len(glm_weights))
n_dims = min(solar_weights.shape[1], glm_weights.shape[1])

solar_weights = solar_weights[:n_layers, :n_dims]
glm_weights = glm_weights[:n_layers, :n_dims]

if intellect_weights is not None:
    intellect_n_layers = min(len(intellect_weights), n_layers)
    intellect_weights = intellect_weights[:intellect_n_layers, :n_dims]

if phi_weights is not None:
    phi_n_layers = min(len(phi_weights), n_layers)
    phi_n_dims = min(phi_weights.shape[1], n_dims)
    phi_weights = phi_weights[:phi_n_layers, :phi_n_dims]

output_dir = os.path.join(script_dir, "result")
os.makedirs(output_dir, exist_ok=True)

fig = plt.figure(figsize=(16, 14))

surface_models = [
    ("Solar", solar_weights, "Purples"),
    ("GLM", glm_weights, "Greys"),
]
if intellect_weights is not None:
    surface_models.append(("INTELLECT", intellect_weights, "Greys"))
if phi_weights is not None:
    surface_models.append(("Phi", phi_weights, "Greens"))

N_SHOW_DIMS = 100
N_SHOW_LAYERS = min(n_layers, 47)

all_max = max(w[:N_SHOW_LAYERS, :N_SHOW_DIMS].max() for _, w, _ in surface_models)
vmax = min(2.0, all_max)

for idx, (name, weights, cmap) in enumerate(surface_models):
    ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
    n_l = min(N_SHOW_LAYERS, len(weights))
    n_d = min(N_SHOW_DIMS, weights.shape[1])
    weights_sub = weights[:n_l, :n_d]

    X, Y = np.meshgrid(np.arange(n_d), np.arange(n_l))
    surf = ax.plot_surface(X, Y, weights_sub, cmap=cmap, alpha=0.9,
                           edgecolor="none", vmin=0, vmax=vmax)

    ax.set_xlabel("Dimension", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_zlabel("γ", fontsize=9)
    ax.set_zlim(0, vmax)
    ax.set_title(f"{name} LayerNorm Weight (γ)", fontweight="bold", fontsize=11)
    ax.view_init(elev=25, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "figure_06_layernorm_surface.png"), dpi=200, bbox_inches="tight")
plt.close()

models_for_pca = [("Solar", solar_weights), ("GLM", glm_weights)]
if intellect_weights is not None:
    models_for_pca.append(("INTELLECT", intellect_weights))
if phi_weights is not None:
    phi_padded = np.zeros((len(phi_weights), n_dims))
    phi_padded[:, :phi_weights.shape[1]] = phi_weights
    models_for_pca.append(("Phi", phi_padded))

min_layers_pca = min(len(w) for _, w in models_for_pca)

combined = np.vstack([w[:min_layers_pca] for _, w in models_for_pca])
pca = PCA(n_components=3)
combined_pca = pca.fit_transform(combined)

pca_results = {}
offset = 0
for name, w in models_for_pca:
    n = min_layers_pca
    pca_results[name] = combined_pca[offset:offset + n]
    offset += n

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

for name, pca_data in pca_results.items():
    color = COLORS[name]
    marker = MARKERS[name]

    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
               c=color, s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
               label=name, marker=marker)

    ax.plot(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
            "-", color=color, alpha=0.4, linewidth=1.5)

    ax.scatter(pca_data[0, 0], pca_data[0, 1], pca_data[0, 2],
               c=color, s=150, marker="o", edgecolors="black", linewidth=2)
    ax.scatter(pca_data[-1, 0], pca_data[-1, 1], pca_data[-1, 2],
               c=color, s=150, marker="*", edgecolors="black", linewidth=1)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("PCA Projection of LayerNorm Weights (4 Models, 3D)\n(each point = one layer, large circle = L0, star = last layer)", fontweight="bold")
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "figure_07_layernorm_pca.png"), dpi=200, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models_stats = [("Solar", solar_weights), ("GLM", glm_weights)]
if intellect_weights is not None:
    models_stats.append(("INTELLECT", intellect_weights))
if phi_weights is not None:
    models_stats.append(("Phi", phi_weights))

ax = axes[0, 0]
for name, weights in models_stats:
    ax.plot(range(len(weights)), weights.mean(axis=1),
            f"-{MARKERS[name]}", color=COLORS[name],
            label=name, markersize=4, alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean weight")
ax.set_title("Mean Weight per Layer", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for name, weights in models_stats:
    ax.plot(range(len(weights)), weights.std(axis=1),
            f"-{MARKERS[name]}", color=COLORS[name],
            label=name, markersize=4, alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Std weight")
ax.set_title("Weight Std per Layer", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for name, weights in models_stats:
    ax.plot(range(len(weights)), weights.min(axis=1),
            f"-{MARKERS[name]}", color=COLORS[name],
            label=name, markersize=4, alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Min weight")
ax.set_title("Min Weight per Layer", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for name, weights in models_stats:
    ax.plot(range(len(weights)), weights.max(axis=1),
            f"-{MARKERS[name]}", color=COLORS[name],
            label=name, markersize=4, alpha=0.8)
ax.set_xlabel("Layer")
ax.set_ylabel("Max weight")
ax.set_title("Max Weight per Layer", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "figure_08_layernorm_stats.png"), dpi=200, bbox_inches="tight")
plt.close()
