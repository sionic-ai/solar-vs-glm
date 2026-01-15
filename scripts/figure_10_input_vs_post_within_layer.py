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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_layernorm_weight(model_id: str, layer_idx: int, norm_type: str = "input"):
    """Load LayerNorm weight tensor."""
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
        return None
    return None


def pearson_correlation(a, b):
    a = a.flatten()
    b = b.flatten()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < 1e-10:
        return np.nan
    return float(np.dot(a_c, b_c) / denom)


def main():
    solar_id = "upstage/Solar-Open-100B"
    glm_id = "zai-org/GLM-4.5-Air"
    max_layer = 46

    means = {"Solar": [], "GLM": []}
    stds = {"Solar": [], "GLM": []}
    corrs = []
    layers = []

    for layer in range(max_layer + 1):
        solar_in = load_layernorm_weight(solar_id, layer, "input")
        solar_post = load_layernorm_weight(solar_id, layer, "post")
        glm_in = load_layernorm_weight(glm_id, layer, "input")
        glm_post = load_layernorm_weight(glm_id, layer, "post")

        if any(x is None for x in (solar_in, solar_post, glm_in, glm_post)):
            continue
        if solar_in.shape != solar_post.shape or glm_in.shape != glm_post.shape:
            continue

        solar_diff = solar_in - solar_post
        glm_diff = glm_in - glm_post

        means["Solar"].append(solar_diff.mean())
        means["GLM"].append(glm_diff.mean())
        stds["Solar"].append(solar_diff.std())
        stds["GLM"].append(glm_diff.std())
        corrs.append(pearson_correlation(solar_diff, glm_diff))
        layers.append(layer)

    if not layers:
        raise SystemExit("No comparable layers found for Solar and GLM.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1 = axes[0]
    ax1.plot(layers, means["Solar"], "-o", color="#8B7FC7", markersize=3, label="Solar")
    ax1.plot(layers, means["GLM"], "-s", color="#4a4a4a", markersize=3, label="GLM")
    ax1.set_title("(a) Mean (input - post)", fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean")
    ax1.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(layers, stds["Solar"], "-o", color="#8B7FC7", markersize=3, label="Solar")
    ax2.plot(layers, stds["GLM"], "-s", color="#4a4a4a", markersize=3, label="GLM")
    ax2.set_title("(b) Std (input - post)", fontweight="bold")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Std")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[2]
    ax3.plot(layers, corrs, "-^", color="#5e3c99", markersize=3)
    ax3.set_title("(c) Pearson (Solar vs GLM)", fontweight="bold")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Pearson r")
    ax3.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax3.set_ylim(-0.2, 0.2)
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Within-Layer Analysis: input_LN vs post_LN", fontweight="bold", y=1.02)
    plt.tight_layout()

    output_dir = os.path.join(SCRIPT_DIR, "result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure_10_input_vs_post_within_layer.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
