#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_comprehensive.py

Comprehensive analysis to prove Solar-Open-100B is derived from GLM-4.5-Air.

Key evidence to collect:
1. LayerNorm weight similarity (should be VERY high if same base)
2. Tokenizer overlap analysis
3. Embedding weight analysis (first rows for shared tokens)
4. Random baseline comparison (what similarity do unrelated models have?)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url

# ============ MODELS ============
MODEL_A = "upstage/Solar-Open-100B"      # Suspected derivative
MODEL_B = "zai-org/GLM-4.5-Air"          # Suspected base

# ============ UTILITIES ============
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def hf_download(repo_id: str, filename: str, revision: str, token: Optional[str]) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)

def load_config(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    path = hf_download(repo_id, "config.json", revision, token)
    return read_json(path)

def load_tokenizer_json(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    path = hf_download(repo_id, "tokenizer.json", revision, token)
    return read_json(path)

def load_index_json(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    path = hf_download(repo_id, "model.safetensors.index.json", revision, token)
    return read_json(path)

# ============ SAFETENSORS ============
def http_range_get(url: str, start: int, end: int, token: Optional[str], retries: int = 5) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
        except:
            pass
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Range GET failed")

def fetch_safetensors_header(repo_id: str, filename: str, revision: str, token: Optional[str], max_header_bytes: int) -> Dict[str, Any]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = int.from_bytes(raw8, "little")
    if header_len > max_header_bytes:
        raise RuntimeError(f"Header too large: {header_len}")
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header

def decode_tensor(raw: bytes, dtype: str, shape: List[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = (u16.astype(np.uint32) << 16)
        arr = u32.view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float32)

    size = int(np.prod(shape)) if shape else 1
    if arr.size != size:
        arr = arr[:size]
    return arr.reshape(shape)

def fetch_tensor(repo_id: str, filename: str, revision: str, key: str, header: Dict[str, Any], token: Optional[str]) -> np.ndarray:
    info = header[key]
    dtype = info["dtype"]
    shape = info["shape"]
    off0, off1 = info["data_offsets"]
    header_len = int(header["__header_len__"])
    url = header["__url__"]
    begin = 8 + header_len + int(off0)
    end_incl = 8 + header_len + int(off1) - 1
    raw = http_range_get(url, begin, end_incl, token)
    return decode_tensor(raw, dtype, shape)

def tensor_nbytes(info: Dict[str, Any]) -> int:
    a, b = info["data_offsets"]
    return int(b) - int(a)

# ============ SIMILARITIES ============
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

# ============ RANDOM BASELINE ============
def generate_random_baseline(dim: int = 4096, n_trials: int = 1000) -> Dict[str, float]:
    """
    Generate random vectors and compute their similarity as baseline.
    This tells us what similarity to expect from completely unrelated vectors.
    """
    cosines = []
    pearsons = []

    for _ in range(n_trials):
        # Random vectors with similar distribution to LayerNorm weights
        a = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)

        # Normalize like typical LayerNorm weights (centered around 1)
        a = a * 0.01 + 1.0
        b = b * 0.01 + 1.0

        cosines.append(cosine(a, b))
        pearsons.append(pearson(a, b))

    return {
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines)),
        "cosine_max": float(np.max(cosines)),
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons)),
    }

# ============ TOKENIZER ANALYSIS ============
def inv_vocab(tok_json: Dict[str, Any]) -> List[str]:
    vocab = tok_json["model"]["vocab"]
    inv = [None] * len(vocab)
    for t, i in vocab.items():
        inv[i] = t
    return inv

def deep_tokenizer_analysis(repoA: str, repoB: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    """
    Deep analysis of tokenizer relationship.
    """
    tjA = load_tokenizer_json(repoA, revision, token)
    tjB = load_tokenizer_json(repoB, revision, token)

    vocabA = tjA["model"]["vocab"]
    vocabB = tjB["model"]["vocab"]
    invA = inv_vocab(tjA)
    invB = inv_vocab(tjB)

    # Find common tokens and their ID mappings
    common_tokens = set(vocabA.keys()) & set(vocabB.keys())

    # Analyze ID shifts
    shifts = []
    for tok in common_tokens:
        idA = vocabA[tok]
        idB = vocabB[tok]
        shifts.append(idA - idB)

    shift_counter = Counter(shifts)
    most_common_shifts = shift_counter.most_common(20)

    # Find the dominant shift
    dominant_shift = most_common_shifts[0][0] if most_common_shifts else 0
    tokens_at_dominant_shift = most_common_shifts[0][1] if most_common_shifts else 0

    # Check if GLM's first N tokens map to Solar with consistent offset
    first_n = min(1000, len(invB))
    consistent_mapping_count = 0
    sample_mappings = []

    for idB in range(first_n):
        tokB = invB[idB]
        if tokB and tokB in vocabA:
            idA = vocabA[tokB]
            if idA == idB + dominant_shift:
                consistent_mapping_count += 1
            if len(sample_mappings) < 20:
                sample_mappings.append({
                    "token": tokB[:30],  # truncate for display
                    "id_in_GLM": idB,
                    "id_in_Solar": idA,
                    "shift": idA - idB
                })

    # Merges analysis
    mergesA = tjA["model"].get("merges", []) or []
    mergesB = tjB["model"].get("merges", []) or []

    def normalize_merge(m):
        if isinstance(m, str):
            return m
        return " ".join(map(str, m))

    setA = set(normalize_merge(m) for m in mergesA)
    setB = set(normalize_merge(m) for m in mergesB)

    merges_only_in_B = setB - setA  # These would indicate B came first
    merges_only_in_A = setA - setB  # These would be additions in A

    return {
        "vocab_size_A": len(vocabA),
        "vocab_size_B": len(vocabB),
        "vocab_diff": len(vocabA) - len(vocabB),
        "common_tokens": len(common_tokens),
        "common_ratio_over_B": len(common_tokens) / len(vocabB) if vocabB else 0,
        "dominant_id_shift": dominant_shift,
        "tokens_at_dominant_shift": tokens_at_dominant_shift,
        "first_1000_consistent_mapping": consistent_mapping_count,
        "first_1000_consistent_ratio": consistent_mapping_count / first_n if first_n else 0,
        "sample_token_mappings": sample_mappings,
        "most_common_shifts": [{"shift": s, "count": c} for s, c in most_common_shifts[:10]],
        "merges_len_A": len(mergesA),
        "merges_len_B": len(mergesB),
        "merges_common": len(setA & setB),
        "merges_only_in_A_count": len(merges_only_in_A),
        "merges_only_in_B_count": len(merges_only_in_B),
        "merges_B_coverage_in_A": len(setA & setB) / len(setB) if setB else 0,
    }

# ============ KEY DISCOVERY ============
LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")

def get_layer_index(key: str) -> Optional[int]:
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None

def classify_key(key: str) -> Optional[str]:
    lk = key.lower()
    if not lk.endswith(".weight"):
        return None

    if any(x in lk for x in ["router", "gate", "gating"]) and ("proj" not in lk):
        return "router_or_gate"

    if "norm" in lk or "layernorm" in lk or "rms" in lk:
        if any(x in lk for x in ["input", "pre", "attn_norm", "attention_norm", "ln_1"]):
            return "norm_pre"
        if any(x in lk for x in ["post", "ffn", "mlp", "ln_2"]):
            return "norm_post"
        return "norm_any"

    return None

def build_layer_keyrefs(weight_map: Dict[str, str], max_layers: int) -> Dict[int, Dict[str, Any]]:
    per_layer = defaultdict(lambda: defaultdict(list))
    for k, shard in weight_map.items():
        li = get_layer_index(k)
        if li is None or li >= max_layers:
            continue
        cat = classify_key(k)
        if cat is None:
            continue
        per_layer[li][cat].append({"key": k, "shard": shard})

    out = {}
    for li, cats in per_layer.items():
        chosen = {}
        for cat, lst in cats.items():
            lst_sorted = sorted(lst, key=lambda x: x["key"])
            chosen[cat] = lst_sorted[0]

        if "norm_pre" not in chosen and "norm_any" in chosen:
            chosen["norm_pre"] = chosen["norm_any"]
        if "norm_post" not in chosen and "norm_any" in chosen:
            chosen["norm_post"] = chosen["norm_any"]

        out[li] = chosen

    return out

# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_comprehensive")
    ap.add_argument("--max-layers", type=int, default=46)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    ensure_outdir(args.outdir)
    revision = "main"
    max_header_bytes = 64 * 1024 * 1024
    max_tensor_bytes = 2_500_000

    print("=" * 70)
    print("COMPREHENSIVE PROVENANCE ANALYSIS")
    print("Solar-Open-100B (A) vs GLM-4.5-Air (B)")
    print("=" * 70)

    # ============ 1. RANDOM BASELINE ============
    print("\n[1/4] Computing random baseline...")
    baseline = generate_random_baseline(dim=4096, n_trials=10000)
    print(f"  Random vectors (dim=4096):")
    print(f"    Cosine:  mean={baseline['cosine_mean']:.4f} +/- {baseline['cosine_std']:.4f}, max={baseline['cosine_max']:.4f}")
    print(f"    Pearson: mean={baseline['pearson_mean']:.4f} +/- {baseline['pearson_std']:.4f}")

    # ============ 2. TOKENIZER ANALYSIS ============
    print("\n[2/4] Deep tokenizer analysis...")
    tok_analysis = deep_tokenizer_analysis(MODEL_A, MODEL_B, revision, token)
    write_json(os.path.join(args.outdir, "tokenizer_analysis.json"), tok_analysis)

    print(f"  Vocab sizes: Solar={tok_analysis['vocab_size_A']}, GLM={tok_analysis['vocab_size_B']}")
    print(f"  Solar has {tok_analysis['vocab_diff']} MORE tokens")
    print(f"  Common tokens: {tok_analysis['common_tokens']} ({tok_analysis['common_ratio_over_B']*100:.1f}% of GLM)")
    print(f"  Dominant ID shift: {tok_analysis['dominant_id_shift']} (affects {tok_analysis['tokens_at_dominant_shift']} tokens)")
    print(f"  First 1000 GLM tokens with consistent mapping: {tok_analysis['first_1000_consistent_ratio']*100:.1f}%")
    print(f"  GLM merges found in Solar: {tok_analysis['merges_B_coverage_in_A']*100:.1f}%")

    # ============ 3. LAYERWISE WEIGHT COMPARISON ============
    print("\n[3/4] Layerwise weight comparison...")

    cfgA = load_config(MODEL_A, revision, token)
    cfgB = load_config(MODEL_B, revision, token)
    idxA = load_index_json(MODEL_A, revision, token)
    idxB = load_index_json(MODEL_B, revision, token)

    wmA = idxA.get("weight_map", {})
    wmB = idxB.get("weight_map", {})

    layersA = int(cfgA.get("num_hidden_layers", 0))
    layersB = int(cfgB.get("num_hidden_layers", 0))
    probe_layers = min(layersA, layersB, args.max_layers)

    refsA = build_layer_keyrefs(wmA, max_layers=layersA)
    refsB = build_layer_keyrefs(wmB, max_layers=layersB)

    common_layers = sorted(set(refsA.keys()) & set(refsB.keys()))
    common_layers = [L for L in common_layers if L < probe_layers]

    header_cache = {}

    def get_header(repo, shard):
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_safetensors_header(repo, shard, revision, token, max_header_bytes)
        return header_cache[k]

    rows = []
    categories = ["norm_pre", "norm_post", "router_or_gate"]

    for layer in common_layers:
        catsA = refsA.get(layer, {})
        catsB = refsB.get(layer, {})

        for cat in categories:
            if cat not in catsA or cat not in catsB:
                continue

            refA = catsA[cat]
            refB = catsB[cat]

            try:
                hA = get_header(MODEL_A, refA["shard"])
                hB = get_header(MODEL_B, refB["shard"])

                if refA["key"] not in hA or refB["key"] not in hB:
                    continue

                bytesA = tensor_nbytes(hA[refA["key"]])
                bytesB = tensor_nbytes(hB[refB["key"]])

                if bytesA > max_tensor_bytes or bytesB > max_tensor_bytes:
                    continue

                A = fetch_tensor(MODEL_A, refA["shard"], revision, refA["key"], hA, token)
                B = fetch_tensor(MODEL_B, refB["shard"], revision, refB["key"], hB, token)

                if A.shape != B.shape and A.ndim == 2 and B.ndim == 2 and A.shape == B.shape[::-1]:
                    B = B.T

                if A.shape != B.shape:
                    continue

                rows.append({
                    "layer": layer,
                    "category": cat,
                    "cosine": cosine(A, B),
                    "pearson": pearson(A, B),
                })

            except Exception as e:
                eprint(f"[warn] layer={layer} cat={cat}: {e}")

    print(f"  Collected {len(rows)} comparison points")

    # ============ 4. STATISTICAL ANALYSIS ============
    print("\n[4/4] Statistical analysis...")

    def calc_stats(rows: List[Dict], category: str) -> Dict[str, float]:
        vals = [r["cosine"] for r in rows if r["category"] == category and not np.isnan(r["cosine"])]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "count": len(vals),
        }

    stats = {
        "norm_pre": calc_stats(rows, "norm_pre"),
        "norm_post": calc_stats(rows, "norm_post"),
        "router_or_gate": calc_stats(rows, "router_or_gate"),
        "random_baseline": baseline,
    }

    write_json(os.path.join(args.outdir, "statistics.json"), stats)

    # ============ PLOTTING ============
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cosine by layer
    ax1 = axes[0, 0]
    for cat, color, label in [("norm_pre", "blue", "input_layernorm"),
                               ("norm_post", "green", "post_attention_layernorm"),
                               ("router_or_gate", "red", "router/gate")]:
        xs = [r["layer"] for r in rows if r["category"] == cat]
        ys = [r["cosine"] for r in rows if r["category"] == cat]
        if xs:
            ax1.plot(xs, ys, 'o-', color=color, label=label, markersize=4, alpha=0.8)

    ax1.axhline(y=baseline["cosine_mean"], color='gray', linestyle='--', label=f'Random baseline ({baseline["cosine_mean"]:.3f})')
    ax1.axhline(y=baseline["cosine_mean"] + 3*baseline["cosine_std"], color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Layerwise Cosine Similarity\nSolar-Open-100B vs GLM-4.5-Air')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.05)

    # Plot 2: Pearson by layer
    ax2 = axes[0, 1]
    for cat, color, label in [("norm_pre", "blue", "input_layernorm"),
                               ("norm_post", "green", "post_attention_layernorm")]:
        xs = [r["layer"] for r in rows if r["category"] == cat]
        ys = [r["pearson"] for r in rows if r["category"] == cat]
        if xs:
            ax2.plot(xs, ys, 'o-', color=color, label=label, markersize=4, alpha=0.8)

    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Layerwise Pearson Correlation\n(Mean-centered comparison)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histogram comparison
    ax3 = axes[1, 0]

    norm_cosines = [r["cosine"] for r in rows if r["category"] in ["norm_pre", "norm_post"]]
    random_samples = [cosine(np.random.randn(4096)*0.01+1, np.random.randn(4096)*0.01+1) for _ in range(1000)]

    ax3.hist(random_samples, bins=50, alpha=0.5, label='Random baseline', color='gray', density=True)
    ax3.hist(norm_cosines, bins=30, alpha=0.7, label='Solar vs GLM norms', color='blue', density=True)
    ax3.axvline(x=np.mean(norm_cosines), color='blue', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(random_samples), color='gray', linestyle='--', linewidth=2)
    ax3.set_xlabel('Cosine Similarity')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution: Solar-GLM vs Random Baseline')
    ax3.legend()

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
EVIDENCE SUMMARY
================

1. WEIGHT SIMILARITY
   - LayerNorm cosine: {stats['norm_pre']['mean']:.4f} (mean), {stats['norm_pre']['max']:.4f} (max)
   - Random baseline:  {baseline['cosine_mean']:.4f} (mean), {baseline['cosine_max']:.4f} (max)
   - Deviation: {(stats['norm_pre']['mean'] - baseline['cosine_mean']) / baseline['cosine_std']:.1f} sigma above random

2. TOKENIZER RELATIONSHIP
   - Solar has {tok_analysis['vocab_diff']:,} MORE tokens than GLM
   - {tok_analysis['common_ratio_over_B']*100:.1f}% of GLM tokens exist in Solar
   - Dominant ID shift: {tok_analysis['dominant_id_shift']}
   - {tok_analysis['merges_B_coverage_in_A']*100:.1f}% of GLM merges exist in Solar

3. INTERPRETATION
   - Cosine ~1.0 with Pearson ~0 suggests:
     * Vectors point in SAME direction (angle preserved)
     * But magnitudes have drifted (continual training)
   - This is CONSISTENT with continual pretraining
   - GLM -> Solar with tokenizer expansion (+{tok_analysis['vocab_diff']:,} tokens)
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "comprehensive_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ============ FINAL REPORT ============
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- WEIGHT SIMILARITY ---")
    print(f"  norm_pre cosine:  mean={stats['norm_pre']['mean']:.4f}, max={stats['norm_pre']['max']:.4f}")
    print(f"  norm_post cosine: mean={stats['norm_post']['mean']:.4f}, max={stats['norm_post']['max']:.4f}")
    print(f"  Random baseline:  mean={baseline['cosine_mean']:.4f}, max={baseline['cosine_max']:.4f}")

    sigma_above = (stats['norm_pre']['mean'] - baseline['cosine_mean']) / baseline['cosine_std']
    print(f"\n  Solar-GLM similarity is {sigma_above:.0f} SIGMA above random!")
    print(f"  P-value: < 10^-{int(min(300, sigma_above**2/2))} (effectively zero)")

    print("\n--- TOKENIZER ---")
    print(f"  Solar = GLM + {tok_analysis['vocab_diff']:,} new tokens")
    print(f"  {tok_analysis['merges_B_coverage_in_A']*100:.1f}% of GLM's BPE merges preserved in Solar")

    print("\n--- CONCLUSION ---")
    if stats['norm_pre']['mean'] > 0.9 and sigma_above > 100:
        print("  [VERY STRONG EVIDENCE] Solar-Open-100B appears to be derived from GLM-4.5-Air")
        print("  - LayerNorm weights show near-identical direction (cosine > 0.9)")
        print("  - Statistical impossibility of random occurrence")
        print("  - Tokenizer shows expansion pattern (GLM subset + additions)")
        print("  - Pattern consistent with CONTINUAL PRETRAINING with tokenizer expansion")
    elif stats['norm_pre']['mean'] > 0.7:
        print("  [STRONG EVIDENCE] High similarity suggests shared lineage")
    else:
        print("  [INCONCLUSIVE] More investigation needed")

    print("\n" + "=" * 70)
    print(f"Full results: {args.outdir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
