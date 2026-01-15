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
    'font.size': 9,
    'axes.titlesize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    'Solar': 'upstage/Solar-Open-100B',
    'GLM': 'zai-org/GLM-4.5-Air',
}

COLORS = {
    'Solar': '#8B7FC7',
    'GLM': '#4a4a4a',
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


def load_all_layers(model_id: str, norm_type: str, max_layer: int = 46):
    """Load weights from all layers."""
    weights = []
    for layer_idx in range(max_layer + 1):
        w = load_layernorm_weight(model_id, layer_idx, norm_type)
        if w is not None:
            weights.append(w)
    return np.array(weights)


        
solar_input = load_all_layers(MODELS['Solar'], "input")
solar_post = load_all_layers(MODELS['Solar'], "post")
glm_input = load_all_layers(MODELS['GLM'], "input")
glm_post = load_all_layers(MODELS['GLM'], "post")

output_dir = os.path.join(script_dir, "result")
os.makedirs(output_dir, exist_ok=True)


def plot_ridge_overlay(solar_weights, glm_weights, title, output_name):
    """Overlay ridge plot for Solar vs GLM."""
    n_layers = min(len(solar_weights), len(glm_weights))

              
    sample_layers = list(range(0, n_layers, 2))             

    fig, ax = plt.subplots(figsize=(10, 12))

    for i, layer_idx in enumerate(sample_layers):
        offset = i * 1.2            

                  
        solar_data = solar_weights[layer_idx]
        solar_hist, solar_bins = np.histogram(solar_data, bins=60, range=(-0.2, 2.0), density=True)
        solar_centers = (solar_bins[:-1] + solar_bins[1:]) / 2

                
        glm_data = glm_weights[layer_idx]
        glm_hist, glm_bins = np.histogram(glm_data, bins=60, range=(-0.2, 2.0), density=True)
        glm_centers = (glm_bins[:-1] + glm_bins[1:]) / 2

        scale = 0.15          

                     
        ax.fill_between(glm_centers, offset, offset + glm_hist * scale,
                        alpha=0.5, color=COLORS['GLM'], label='GLM' if i == 0 else '')
        ax.plot(glm_centers, offset + glm_hist * scale, color=COLORS['GLM'], linewidth=0.8)

                    
        ax.fill_between(solar_centers, offset, offset + solar_hist * scale,
                        alpha=0.5, color=COLORS['Solar'], label='Solar' if i == 0 else '')
        ax.plot(solar_centers, offset + solar_hist * scale, color=COLORS['Solar'], linewidth=0.8)

                
        ax.text(-0.35, offset + 0.15, f'Layer {layer_idx}', fontsize=8, ha='right', va='center')

    ax.set_xlabel('Weight value (γ)', fontsize=10)
    ax.set_ylabel('Layer', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(-0.4, 2.0)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.3, label='init=1.0')

                   
    ax.text(1.02, ax.get_ylim()[1] * 0.95, '1.0', fontsize=8, color='red', alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


                

n_layers = min(len(solar_input), len(glm_input))
sample_layers = list(range(0, n_layers, 3))             

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, (solar_w, glm_w, title) in zip(axes, [
    (solar_input, glm_input, 'input_layernorm'),
    (solar_post, glm_post, 'post_attention_layernorm')
]):
    for i, layer_idx in enumerate(sample_layers):
        offset = i * 1.0

                  
        solar_data = solar_w[layer_idx]
        solar_hist, solar_bins = np.histogram(solar_data, bins=60, range=(-0.2, 2.0), density=True)
        solar_centers = (solar_bins[:-1] + solar_bins[1:]) / 2

                
        glm_data = glm_w[layer_idx]
        glm_hist, glm_bins = np.histogram(glm_data, bins=60, range=(-0.2, 2.0), density=True)
        glm_centers = (glm_bins[:-1] + glm_bins[1:]) / 2

        scale = 0.12

                 
        ax.fill_between(glm_centers, offset, offset + glm_hist * scale,
                        alpha=0.5, color=COLORS['GLM'], label='GLM' if i == 0 else '')
        ax.plot(glm_centers, offset + glm_hist * scale, color=COLORS['GLM'], linewidth=0.8)

                   
        ax.fill_between(solar_centers, offset, offset + solar_hist * scale,
                        alpha=0.5, color=COLORS['Solar'], label='Solar' if i == 0 else '')
        ax.plot(solar_centers, offset + solar_hist * scale, color=COLORS['Solar'], linewidth=0.8)

                
        ax.text(-0.3, offset + 0.1, f'L{layer_idx}', fontsize=7, ha='right', va='center')

    ax.set_xlabel('Weight value (γ)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.4, 2.0)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=9)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.3)

plt.suptitle('LayerNorm Weight Distribution by Layer', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
output_path = os.path.join(output_dir, "figure_09_ridge_combined.png")
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
