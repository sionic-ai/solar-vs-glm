#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "huggingface_hub",
#     "safetensors",
#     "torch",
# ]
# ///

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from safetensors import safe_open

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_layernorm_weight(model_id: str, layer_idx: int, norm_type: str = "input"):
    """LayerNorm weight 로드"""
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
    except Exception as e:
        pass
    return None


def raw_cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pearson_correlation(a, b):
    a, b = a.flatten(), b.flatten()
    a_c, b_c = a - a.mean(), b - b.mean()
    norm_a, norm_b = np.linalg.norm(a_c), np.linalg.norm(b_c)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float('nan')
    return float(np.dot(a_c, b_c) / (norm_a * norm_b))


def main():
    solar_id = "upstage/Solar-Open-100B"
    glm_id = "zai-org/GLM-4.5-Air"

    max_layer = 46
    n_layers = max_layer + 1

    solar_weights = []
    for layer in range(n_layers):
        w = load_layernorm_weight(solar_id, layer, "input")
        solar_weights.append(w)

    glm_weights = []
    for layer in range(n_layers):
        w = load_layernorm_weight(glm_id, layer, "input")
        glm_weights.append(w)
    cosine_matrix = np.full((n_layers, n_layers), np.nan)
    pearson_matrix = np.full((n_layers, n_layers), np.nan)

    for i in range(n_layers):
        for j in range(n_layers):
            if solar_weights[i] is not None and glm_weights[j] is not None:
                if solar_weights[i].shape == glm_weights[j].shape:
                    cosine_matrix[i, j] = raw_cosine(solar_weights[i], glm_weights[j])
                    pearson_matrix[i, j] = pearson_correlation(solar_weights[i], glm_weights[j])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))


    ax1 = axes[0]
    im1 = ax1.imshow(cosine_matrix, cmap='Reds', aspect='equal', vmin=0.9, vmax=1.0)
    ax1.set_xlabel('GLM Layer')
    ax1.set_ylabel('Solar Layer')
    ax1.set_title('(a) Raw Cosine Similarity', fontsize=11, fontweight='bold', pad=10)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Cosine')


    for i in range(n_layers):
        ax1.plot(i, i, 'k.', markersize=2)


    ax2 = axes[1]
    im2 = ax2.imshow(pearson_matrix, cmap='RdBu_r', aspect='equal', vmin=-0.1, vmax=0.1)
    ax2.set_xlabel('GLM Layer')
    ax2.set_ylabel('Solar Layer')
    ax2.set_title('(b) Pearson Correlation', fontsize=11, fontweight='bold', pad=10)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Pearson r')


    for i in range(n_layers):
        ax2.plot(i, i, 'k.', markersize=2)

    fig.suptitle('Solar[i] vs GLM[j] - All Layer Pairs', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(script_dir, "result", "figure_02_heatmap_all_layers.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
