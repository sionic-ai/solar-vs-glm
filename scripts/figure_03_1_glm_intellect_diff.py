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


def main():
    glm_id = "zai-org/GLM-4.5-Air"
    intellect_id = "PrimeIntellect/INTELLECT-3"
    max_layer = 46

    layer_indices = []
    changed_ratio = []
    mean_abs_diff = []

    for layer in range(max_layer + 1):
        glm_w = load_layernorm_weight(glm_id, layer, "input")
        intel_w = load_layernorm_weight(intellect_id, layer, "input")
        if glm_w is None or intel_w is None:
            continue
        if glm_w.shape != intel_w.shape:
            continue

        diff = (glm_w != intel_w).float().mean().item() * 100.0
        mean_diff = (glm_w.float() - intel_w.float()).abs().mean().item()

        layer_indices.append(layer)
        changed_ratio.append(diff)
        mean_abs_diff.append(mean_diff)

    if not layer_indices:
        raise SystemExit("No comparable layers found for GLM and INTELLECT.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.bar(layer_indices, changed_ratio, color="#d95f02", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Changed ratio (%)")
    ax1.set_title("(a) Ratio of changed values", fontweight="bold")
    ax1.set_ylim(0, max(changed_ratio) * 1.1)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.plot(layer_indices, mean_abs_diff, "-o", color="#1b9e77", markersize=3, linewidth=1.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean |diff|")
    ax2.set_title("(b) Mean absolute diff (log scale)", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("GLM vs INTELLECT LayerNorm Diff Analysis", fontweight="bold", y=1.02)
    plt.tight_layout()

    output_dir = os.path.join(SCRIPT_DIR, "result")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure_03_1_glm_intellect_diff_analysis.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
