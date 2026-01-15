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
from matplotlib.colors import LinearSegmentedColormap
from huggingface_hub import hf_hub_download
from safetensors import safe_open

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

           
COLORS = {
    'Solar': '#8B7FC7',          
    'GLM': '#4a4a4a',               
    'INTELLECT': '#9a9a9a',         
    'Phi': '#2ca02c',            
}


def hex_to_rgb(hex_color):
    """Hex to RGB (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))


def create_pair_colormap(model_a, model_b, is_derivative=False):
    """Create custom colormap for model pair."""
    color_a = hex_to_rgb(COLORS[model_a])
    color_b = hex_to_rgb(COLORS[model_b])

                       
    blend = tuple((a + b) / 2 for a, b in zip(color_a, color_b))

    if is_derivative:
                                      
        return LinearSegmentedColormap.from_list(
            f'{model_a}_{model_b}',
            ['white', blend, blend],
            N=256
        )
    else:
                                  
        light_blend = tuple(0.3 + 0.7 * c for c in blend)         
        return LinearSegmentedColormap.from_list(
            f'{model_a}_{model_b}',
            [light_blend, 'white', light_blend],
            N=256
        )

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    'Solar': 'upstage/Solar-Open-100B',
    'GLM': 'zai-org/GLM-4.5-Air',
    'INTELLECT': 'PrimeIntellect/INTELLECT-3',
    'Phi': 'microsoft/Phi-3.5-MoE-instruct',
}


def load_layernorm_weight(model_id: str, layer_idx: int, norm_type: str = "input"):
    """Load LayerNorm weight."""
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


def load_all_layers(model_id: str, norm_type: str = "input", max_layer: int = 46):
    """Load weights from all layers."""
    weights = []
    for layer_idx in range(max_layer + 1):
        w = load_layernorm_weight(model_id, layer_idx, norm_type)
        if w is not None:
            weights.append(w)
    return weights


def pearson_correlation(a, b):
    a, b = a.flatten(), b.flatten()
    a_c, b_c = a - a.mean(), b - b.mean()
    norm_a, norm_b = np.linalg.norm(a_c), np.linalg.norm(b_c)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float('nan')
    return float(np.dot(a_c, b_c) / (norm_a * norm_b))


def compute_pairwise_matrix(weights_a, weights_b):
    """Compute Pearson correlation for all layer pairs between two models."""
    n_a, n_b = len(weights_a), len(weights_b)
    matrix = np.full((n_a, n_b), np.nan)

    for i in range(n_a):
        for j in range(n_b):
            if weights_a[i] is not None and weights_b[j] is not None:
                                   
                min_dim = min(len(weights_a[i]), len(weights_b[j]))
                matrix[i, j] = pearson_correlation(
                    weights_a[i][:min_dim],
                    weights_b[j][:min_dim]
                )
    return matrix


def main():
    all_weights = {}
    for name, model_id in MODELS.items():
        all_weights[name] = load_all_layers(model_id, "input")

                 
    pairs = [
        ('Solar', 'GLM'),
        ('GLM', 'INTELLECT'),
        ('Solar', 'INTELLECT'),
        ('Solar', 'Phi'),
        ('GLM', 'Phi'),
        ('INTELLECT', 'Phi'),
    ]

    matrices = {}
    stats = {}
    for name_a, name_b in pairs:
        matrix = compute_pairwise_matrix(all_weights[name_a], all_weights[name_b])
        matrices[(name_a, name_b)] = matrix

                               
        n_diag = min(matrix.shape[0], matrix.shape[1])
        diagonal = [matrix[i, i] for i in range(n_diag) if not np.isnan(matrix[i, i])]
        off_diagonal = matrix[~np.eye(matrix.shape[0], matrix.shape[1], dtype=bool)]
        off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

        stats[(name_a, name_b)] = {
            'diag_mean': np.mean(diagonal) if diagonal else np.nan,
            'diag_std': np.std(diagonal) if diagonal else np.nan,
            'off_mean': np.mean(off_diagonal) if len(off_diagonal) > 0 else np.nan,
            'off_std': np.std(off_diagonal) if len(off_diagonal) > 0 else np.nan,
        }

              
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name_a, name_b) in enumerate(pairs):
        ax = axes[idx]
        matrix = matrices[(name_a, name_b)]
        s = stats[(name_a, name_b)]

                                  
        is_derivative = (name_a == 'GLM' and name_b == 'INTELLECT')
        cmap = create_pair_colormap(name_a, name_b, is_derivative)

        if is_derivative:
                                    
            im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.9, vmax=1.0)
        else:
                         
            im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-0.1, vmax=0.1)

        ax.set_xlabel(f'{name_b} Layer')
        ax.set_ylabel(f'{name_a} Layer')

                       
        color_a = COLORS[name_a]
        color_b = COLORS[name_b]
        title = f'{name_a} vs {name_b}'
        ax.set_title(f'{title}\n(diag: r={s["diag_mean"]:.3f})', fontweight='bold')

        plt.colorbar(im, ax=ax, shrink=0.8)

                
        n_diag = min(matrix.shape[0], matrix.shape[1])
        for i in range(n_diag):
            ax.plot(i, i, 'k.', markersize=1)

    fig.suptitle('Pearson Correlation Heatmaps: All Model Pairs\n(Layer-wise comparison)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(script_dir, "result", "figure_05_heatmap_multi_pairs.png")
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
