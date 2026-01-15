#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_layer_decay.py

Focused analysis on LAYER DECAY PATTERN - key evidence for continual pretraining.

If Solar is continual pretrained from GLM:
- Early layers should show STRONG similarity (close to original)
- Later layers should show WEAKER similarity (drifted during training)
- This creates a "decay" pattern from early to late layers
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url

MODEL_A = "upstage/Solar-Open-100B"
MODEL_B = "zai-org/GLM-4.5-Air"

def hf_download(repo_id: str, filename: str, revision: str, token: Optional[str]) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def http_range_get(url: str, start: int, end: int, token: Optional[str], retries: int = 5) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
        except:
            pass
        time.sleep(1)
    raise RuntimeError("Range GET failed")

def fetch_header(repo_id: str, filename: str, revision: str, token: Optional[str], max_bytes: int) -> Dict:
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = int.from_bytes(raw8, "little")
    if header_len > max_bytes:
        raise RuntimeError("Header too large")
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header

def decode_tensor(raw: bytes, dtype: str, shape: List[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
    return arr[:int(np.prod(shape))].reshape(shape)

def fetch_tensor(repo: str, filename: str, rev: str, key: str, header: Dict, token: Optional[str]) -> np.ndarray:
    info = header[key]
    off0, off1 = info["data_offsets"]
    begin = 8 + header["__header_len__"] + int(off0)
    end = 8 + header["__header_len__"] + int(off1) - 1
    raw = http_range_get(header["__url__"], begin, end, token)
    return decode_tensor(raw, info["dtype"], info["shape"])

def cosine(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def pearson(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

LAYER_RE = re.compile(r"layers\.(\d+)\.")

def classify_key(key: str) -> Optional[str]:
    lk = key.lower()
    if not lk.endswith(".weight"):
        return None
    if "norm" in lk or "rms" in lk:
        if any(x in lk for x in ["input", "pre", "attn"]):
            return "norm_pre"
        if any(x in lk for x in ["post", "ffn", "mlp"]):
            return "norm_post"
    return None

def build_refs(wm: Dict[str, str], max_layers: int) -> Dict[int, Dict[str, Any]]:
    per_layer = defaultdict(lambda: defaultdict(list))
    for k, shard in wm.items():
        m = LAYER_RE.search(k)
        if not m:
            continue
        layer = int(m.group(1))
        if layer >= max_layers:
            continue
        cat = classify_key(k)
        if cat:
            per_layer[layer][cat].append({"key": k, "shard": shard})

    out = {}
    for layer, cats in per_layer.items():
        chosen = {}
        for cat, lst in cats.items():
            chosen[cat] = sorted(lst, key=lambda x: x["key"])[0]
        out[layer] = chosen
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_decay")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.outdir, exist_ok=True)
    rev = "main"
    max_header = 64 * 1024 * 1024

    print("=" * 70)
    print("LAYER DECAY ANALYSIS: Solar-Open-100B vs GLM-4.5-Air")
    print("=" * 70)

    # Load indices
    cfgA = load_json(hf_download(MODEL_A, "config.json", rev, token))
    cfgB = load_json(hf_download(MODEL_B, "config.json", rev, token))
    idxA = load_json(hf_download(MODEL_A, "model.safetensors.index.json", rev, token))
    idxB = load_json(hf_download(MODEL_B, "model.safetensors.index.json", rev, token))

    layersA = cfgA["num_hidden_layers"]
    layersB = cfgB["num_hidden_layers"]
    probe_layers = min(layersA, layersB)

    refsA = build_refs(idxA["weight_map"], layersA)
    refsB = build_refs(idxB["weight_map"], layersB)

    header_cache = {}
    def get_header(repo, shard):
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_header(repo, shard, rev, token, max_header)
        return header_cache[k]

    # Collect tensors
    print("\nCollecting LayerNorm weights...")
    tensorsA = {}  # (layer, cat) -> tensor
    tensorsB = {}

    for layer in range(probe_layers):
        for cat in ["norm_pre", "norm_post"]:
            if cat not in refsA.get(layer, {}) or cat not in refsB.get(layer, {}):
                continue
            try:
                refA, refB = refsA[layer][cat], refsB[layer][cat]
                hA = get_header(MODEL_A, refA["shard"])
                hB = get_header(MODEL_B, refB["shard"])
                A = fetch_tensor(MODEL_A, refA["shard"], rev, refA["key"], hA, token)
                B = fetch_tensor(MODEL_B, refB["shard"], rev, refB["key"], hB, token)
                if A.shape == B.shape:
                    tensorsA[(layer, cat)] = A
                    tensorsB[(layer, cat)] = B
            except Exception as e:
                print(f"  [skip] layer {layer} {cat}: {e}")

    print(f"  Collected {len(tensorsA)} tensor pairs")

    # Compute pairwise similarity matrix (Solar layers vs GLM layers)
    # This will show if Solar[i] is most similar to GLM[i] (diagonal dominance)
    print("\nComputing similarity matrix...")

    layers_available = sorted(set(l for l, c in tensorsA.keys()))
    n = len(layers_available)

    # Matrix: sim_matrix[i][j] = similarity of Solar layer i to GLM layer j
    sim_matrix = np.zeros((n, n))
    for i, li in enumerate(layers_available):
        for j, lj in enumerate(layers_available):
            sims = []
            for cat in ["norm_pre", "norm_post"]:
                if (li, cat) in tensorsA and (lj, cat) in tensorsB:
                    A = tensorsA[(li, cat)]
                    B = tensorsB[(lj, cat)]
                    sims.append(cosine(A, B))
            if sims:
                sim_matrix[i, j] = np.mean(sims)

    # Compute diagonal vs off-diagonal statistics
    diag = np.diag(sim_matrix)
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]

    print(f"\n  Diagonal (matched layers) mean:     {np.mean(diag):.4f}")
    print(f"  Off-diagonal (mismatched) mean:     {np.mean(off_diag):.4f}")
    print(f"  Diagonal dominance:                 {np.mean(diag) - np.mean(off_diag):.4f}")

    # Layer-by-layer similarity (diagonal elements)
    layerwise_sim = list(zip(layers_available, diag))

    # Compute Pearson correlations
    pearson_by_layer = []
    for layer in layers_available:
        ps = []
        for cat in ["norm_pre", "norm_post"]:
            if (layer, cat) in tensorsA and (layer, cat) in tensorsB:
                ps.append(pearson(tensorsA[(layer, cat)], tensorsB[(layer, cat)]))
        pearson_by_layer.append(np.mean(ps) if ps else 0)

    # PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Similarity matrix heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(sim_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)
    ax1.set_xlabel('GLM Layer')
    ax1.set_ylabel('Solar Layer')
    ax1.set_title('Similarity Matrix: Solar[i] vs GLM[j]\n(Diagonal dominance = evidence of same origin)')
    plt.colorbar(im, ax=ax1, label='Cosine Similarity')
    # Add diagonal line
    ax1.plot([0, n-1], [0, n-1], 'r--', linewidth=2, alpha=0.5)

    # Plot 2: Diagonal similarity by layer (decay pattern)
    ax2 = axes[0, 1]
    ax2.plot(layers_available, diag, 'b-o', linewidth=2, markersize=6, label='Matched (diagonal)')
    ax2.axhline(np.mean(off_diag), color='red', linestyle='--', label=f'Mismatched avg ({np.mean(off_diag):.3f})')
    ax2.fill_between(layers_available, np.mean(off_diag), diag, alpha=0.3, color='blue')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Layer Decay Pattern: Solar[i] vs GLM[i]\n(Blue area = evidence above baseline)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.01)

    # Plot 3: Pearson by layer
    ax3 = axes[1, 0]
    ax3.bar(layers_available, pearson_by_layer, color='green', alpha=0.7)
    ax3.axhline(0, color='gray', linestyle='-')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Pearson Correlation')
    ax3.set_title('Pearson Correlation by Layer\n(Near 0 = magnitudes drifted, directions preserved)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Statistics for early vs late layers
    mid = n // 2
    early_layers = layers_available[:mid]
    late_layers = layers_available[mid:]
    early_sim = np.mean([sim_matrix[i, i] for i in range(mid)])
    late_sim = np.mean([sim_matrix[i, i] for i in range(mid, n)])

    summary = f"""
LAYER DECAY ANALYSIS RESULTS
============================

1. SIMILARITY MATRIX STATISTICS
   Diagonal (matched Solar[i] vs GLM[i]):  {np.mean(diag):.4f} ± {np.std(diag):.4f}
   Off-diagonal (mismatched):              {np.mean(off_diag):.4f} ± {np.std(off_diag):.4f}

   DIAGONAL DOMINANCE: {np.mean(diag) - np.mean(off_diag):.4f}
   (Positive = Solar[i] most similar to GLM[i])

2. EARLY vs LATE LAYER COMPARISON
   Early layers (0-{mid-1}):   {early_sim:.4f}
   Late layers ({mid}-{n-1}):    {late_sim:.4f}

   DECAY: {early_sim - late_sim:.4f}
   (Positive = early layers more similar than late)

3. PEARSON CORRELATION
   Mean across all layers: {np.mean(pearson_by_layer):.4f}

   Near-zero Pearson with high cosine means:
   → Vectors point in SAME direction
   → But have DIFFERENT magnitudes
   → Consistent with CONTINUAL PRETRAINING drift

4. INTERPRETATION
"""

    if np.mean(diag) > np.mean(off_diag) + 0.01:
        evidence = "DIAGONAL DOMINANCE detected"
        if early_sim > late_sim + 0.01:
            evidence += "\nEARLY > LATE layer pattern detected"
            evidence += "\n\n→ STRONG EVIDENCE of continual pretraining"
            evidence += "\n→ Solar appears derived from GLM"
        else:
            evidence += "\n→ MODERATE EVIDENCE of shared origin"
    else:
        evidence = "No clear diagonal dominance\n→ WEAK/NO EVIDENCE"

    ax4.text(0.02, 0.98, summary + evidence, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "layer_decay_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Save data
    write_json(os.path.join(args.outdir, "results.json"), {
        "diagonal_mean": float(np.mean(diag)),
        "off_diagonal_mean": float(np.mean(off_diag)),
        "diagonal_dominance": float(np.mean(diag) - np.mean(off_diag)),
        "early_layers_sim": float(early_sim),
        "late_layers_sim": float(late_sim),
        "decay": float(early_sim - late_sim),
        "pearson_mean": float(np.mean(pearson_by_layer)),
        "layerwise_similarity": [{"layer": int(l), "sim": float(s)} for l, s in layerwise_sim],
    })

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nDiagonal (matched):     {np.mean(diag):.4f}")
    print(f"Off-diagonal:           {np.mean(off_diag):.4f}")
    print(f"DIAGONAL DOMINANCE:     {np.mean(diag) - np.mean(off_diag):.4f}")
    print(f"\nEarly layers (0-{mid-1}):  {early_sim:.4f}")
    print(f"Late layers ({mid}-{n-1}):   {late_sim:.4f}")
    print(f"DECAY:                  {early_sim - late_sim:.4f}")

    if np.mean(diag) > np.mean(off_diag) + 0.01 and early_sim > late_sim:
        print("\n[VERDICT] STRONG EVIDENCE: Solar derived from GLM via continual pretraining")
    elif np.mean(diag) > np.mean(off_diag):
        print("\n[VERDICT] MODERATE EVIDENCE: Shared origin likely")
    else:
        print("\n[VERDICT] WEAK EVIDENCE: Inconclusive")

    print(f"\nPlot: {plot_path}")

if __name__ == "__main__":
    main()
