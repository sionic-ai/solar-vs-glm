#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
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
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    'Solar': '#8B7FC7',
    'GLM': '#3d3d3d',
    'INTELLECT': '#7a7a7a',
    'Phi': '#2ca02c',
}

MODELS = {
    'Solar': 'upstage/Solar-Open-100B',
    'GLM': 'zai-org/GLM-4.5-Air',
    'INTELLECT': 'PrimeIntellect/INTELLECT-3',
    'Phi': 'microsoft/Phi-3.5-MoE-instruct',
}


weight_cache = {}


def load_layernorm_weight(model_id: str, layer_idx: int, norm_type: str = "input"):
    cache_key = (model_id, layer_idx, norm_type)
    if cache_key in weight_cache:
        return weight_cache[cache_key]

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
                w = f.get_tensor(key).float().numpy()
                weight_cache[cache_key] = w
                return w
    except Exception:
        pass
    return None


def raw_cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def pearson_correlation(a, b):
    a, b = a.flatten(), b.flatten()
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom == 0:
        return 1.0 if np.allclose(a_c, b_c) else 0.0
    return np.dot(a_c, b_c) / denom


def generate_comparison(model_a, model_b, output_name, title_prefix):
    """모델 쌍 비교 시각화 생성"""

    model_a_id = MODELS[model_a]
    model_b_id = MODELS[model_b]


    max_layer_a = 46
    max_layer_b = 46


    all_cos = []
    all_pear = []
    valid_layers = []

    for layer_idx in range(max(max_layer_a, max_layer_b) + 1):
        wa = load_layernorm_weight(model_a_id, layer_idx, "input")
        wb = load_layernorm_weight(model_b_id, layer_idx, "input")
        if wa is not None and wb is not None and wa.shape == wb.shape:
            c = raw_cosine(wa, wb)
            p = pearson_correlation(wa, wb)
            all_cos.append(c)
            all_pear.append(p)
            valid_layers.append(layer_idx)

    if not valid_layers:
        return


    best_idx = np.argmax(all_cos)
    best_layer = valid_layers[best_idx]
    best_cos = all_cos[best_idx]
    best_pear = all_pear[best_idx]


    wa = load_layernorm_weight(model_a_id, best_layer, "input")
    wb = load_layernorm_weight(model_b_id, best_layer, "input")


    n_samples = 50
    combined = (wa + wb) / 2
    quartiles = np.percentile(combined, [0, 25, 50, 75, 100])

    sampled_indices = []
    np.random.seed(42)
    for i in range(4):
        q_low, q_high = quartiles[i], quartiles[i+1]
        mask = (combined >= q_low) & (combined <= q_high)
        indices_in_quartile = np.where(mask)[0]
        n_from_quartile = n_samples // 4 + (1 if i < n_samples % 4 else 0)
        sampled = np.random.choice(indices_in_quartile, min(n_from_quartile, len(indices_in_quartile)), replace=False)
        sampled_indices.extend(sampled)

    sampled_indices = sorted(sampled_indices[:n_samples])

    wa_sampled = wa[sampled_indices]
    wb_sampled = wb[sampled_indices]

    cos_val = raw_cosine(wa, wb)
    pear_val = pearson_correlation(wa, wb)


    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    x = np.arange(len(sampled_indices))
    width = 0.35


    ax1 = axes[0]
    ax1.bar(x - width/2, wa_sampled, width, label=model_a, color=COLORS[model_a], alpha=0.8)
    ax1.bar(x + width/2, wb_sampled, width, label=model_b, color=COLORS[model_b], alpha=0.8)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Init (1.0)')
    ax1.set_xlabel('Sampled Dimension')
    ax1.set_ylabel('γ value')
    ax1.set_xticks([])
    ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold')
    ax1.text(0.02, 0.98, f'cos = {cos_val:.2f}', transform=ax1.transAxes,
             ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.5))
    ax1.set_title(f'Layer {best_layer}', fontsize=10, fontweight='bold', pad=8)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', fontsize=7)


    ax2 = axes[1]
    wa_centered = wa_sampled - wa_sampled.mean()
    wb_centered = wb_sampled - wb_sampled.mean()
    ax2.bar(x - width/2, wa_centered, width, label=model_a, color=COLORS[model_a], alpha=0.8)
    ax2.bar(x + width/2, wb_centered, width, label=model_b, color=COLORS[model_b], alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Sampled Dimension')
    ax2.set_ylabel('γ - mean(self)')
    ax2.set_xticks([])
    ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold')
    ax2.text(0.02, 0.98, f'r = {pear_val:.2f}', transform=ax2.transAxes,
             ha='left', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.5))
    ax2.set_title('Centered', fontsize=10, fontweight='bold', pad=8)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', fontsize=7)


    ax3 = axes[2]
    ax3.plot(valid_layers, all_cos, '-o', color='#5a5a5a', label='Raw Cosine', markersize=3, linewidth=1.5)
    ax3.plot(valid_layers, all_pear, '-^', color='#8B7FC7', label='Pearson', markersize=3, linewidth=1.5)
    ax3.axvline(x=best_layer, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Similarity')
    ax3.text(-0.12, 1.05, '(c)', transform=ax3.transAxes, fontsize=11, fontweight='bold')
    ax3.set_title('Per-Layer', fontsize=10, fontweight='bold', pad=8)
    ax3.legend(loc='center right', frameon=True, facecolor='white', fontsize=7)
    ax3.set_ylim(-0.2, 1.1)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix}: {model_a} vs {model_b}', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(script_dir, "result", output_name)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()



pairs = [
    ('Solar', 'GLM', 'figure_01_intuition_independent_input.png', 'Independent'),
    ('GLM', 'INTELLECT', 'figure_03_intuition_derivative_input.png', 'Derivative'),
    ('Phi', 'Solar', 'figure_04a_intuition_phi_solar_input.png', 'Control'),
    ('GLM', 'Phi', 'figure_04b_intuition_glm_phi_input.png', 'Control'),
]

for model_a, model_b, output_name, title_prefix in pairs:
    generate_comparison(model_a, model_b, output_name, title_prefix)
