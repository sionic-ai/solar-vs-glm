#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_with_control.py

Compare Solar-Open-100B vs GLM-4.5-Air with a CONTROL model (Qwen2-57B-A14B)
to show baseline similarity levels.

Also analyzes tokenizer expansion evidence.
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

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except:
    HAVE_SCIPY = False

# ============ MODELS TO COMPARE ============
MODEL_A = "upstage/Solar-Open-100B"      # Suspected derivative
MODEL_B = "zai-org/GLM-4.5-Air"          # Suspected base
MODEL_CONTROL = "deepseek-ai/DeepSeek-V2-Lite"  # Unrelated MoE control

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

# ============ TOKENIZER ANALYSIS ============
def inv_vocab(tok_json: Dict[str, Any]) -> List[str]:
    vocab = tok_json["model"]["vocab"]
    inv = [None] * len(vocab)
    for t, i in vocab.items():
        inv[i] = t
    return inv

def analyze_tokenizer_expansion(repoA: str, repoB: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    """
    Analyze if A's tokenizer is an expansion of B's tokenizer.
    Key evidence:
    1. A has more tokens than B
    2. B's tokens are a subset of A's tokens at similar positions
    3. Merges overlap significantly
    """
    tjA = load_tokenizer_json(repoA, revision, token)
    tjB = load_tokenizer_json(repoB, revision, token)

    invA = inv_vocab(tjA)
    invB = inv_vocab(tjB)

    vocabA = tjA["model"]["vocab"]
    vocabB = tjB["model"]["vocab"]

    # Check if B's tokens exist in A
    b_tokens_in_a = 0
    b_tokens_same_or_close_id = 0
    position_diffs = []

    for tok, idB in vocabB.items():
        if tok in vocabA:
            b_tokens_in_a += 1
            idA = vocabA[tok]
            diff = idA - idB
            position_diffs.append(diff)
            if abs(diff) <= 100:  # Close enough
                b_tokens_same_or_close_id += 1

    # Analyze position differences
    diff_counter = Counter(position_diffs)
    most_common_diffs = diff_counter.most_common(20)

    # Merges analysis
    mergesA = tjA["model"].get("merges", []) or []
    mergesB = tjB["model"].get("merges", []) or []

    def normalize_merge(m):
        if isinstance(m, str):
            return m
        return " ".join(map(str, m))

    setA = set(normalize_merge(m) for m in mergesA)
    setB = set(normalize_merge(m) for m in mergesB)

    merges_intersection = len(setA & setB)
    merges_b_in_a = sum(1 for m in setB if m in setA)

    # Check first N merges overlap (order matters for BPE)
    first_n = min(1000, len(mergesA), len(mergesB))
    first_n_match = sum(1 for i in range(first_n)
                       if normalize_merge(mergesA[i]) == normalize_merge(mergesB[i]))

    return {
        "vocab_size_A": len(invA),
        "vocab_size_B": len(invB),
        "vocab_diff": len(invA) - len(invB),
        "b_tokens_found_in_a": b_tokens_in_a,
        "b_tokens_found_in_a_ratio": b_tokens_in_a / len(vocabB) if vocabB else 0,
        "b_tokens_same_or_close_id": b_tokens_same_or_close_id,
        "most_common_position_diffs": most_common_diffs[:10],
        "merges_len_A": len(mergesA),
        "merges_len_B": len(mergesB),
        "merges_intersection": merges_intersection,
        "merges_b_in_a": merges_b_in_a,
        "merges_b_in_a_ratio": merges_b_in_a / len(setB) if setB else 0,
        "first_1000_merges_exact_match": first_n_match,
        "first_1000_merges_exact_match_ratio": first_n_match / first_n if first_n else 0,
    }

# ============ SAFETENSORS REMOTE ============
def http_range_get(url: str, start: int, end: int, token: Optional[str], retries: int = 5) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Range GET failed: {last_err}")

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

# ============ PAIRWISE COMPARISON ============
def compare_models_layerwise(
    repoA: str, repoB: str,
    revision: str, token: Optional[str],
    max_layers: int = 46,
    max_tensor_bytes: int = 2_500_000,
    max_header_mb: int = 64
) -> List[Dict[str, Any]]:
    """
    Compare two models layer by layer, return list of similarity rows.
    """
    max_header_bytes = max_header_mb * 1024 * 1024

    try:
        cfgA = load_config(repoA, revision, token)
        cfgB = load_config(repoB, revision, token)
        idxA = load_index_json(repoA, revision, token)
        idxB = load_index_json(repoB, revision, token)
    except Exception as e:
        eprint(f"Failed to load configs/index for {repoA} vs {repoB}: {e}")
        return []

    wmA = idxA.get("weight_map", {})
    wmB = idxB.get("weight_map", {})

    layersA = int(cfgA.get("num_hidden_layers", 0))
    layersB = int(cfgB.get("num_hidden_layers", 0))

    probe_layers = min(layersA, layersB, max_layers)

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
                hA = get_header(repoA, refA["shard"])
                hB = get_header(repoB, refB["shard"])

                if refA["key"] not in hA or refB["key"] not in hB:
                    continue

                bytesA = tensor_nbytes(hA[refA["key"]])
                bytesB = tensor_nbytes(hB[refB["key"]])

                if bytesA > max_tensor_bytes or bytesB > max_tensor_bytes:
                    continue

                A = fetch_tensor(repoA, refA["shard"], revision, refA["key"], hA, token)
                B = fetch_tensor(repoB, refB["shard"], revision, refB["key"], hB, token)

                # Handle transpose
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
                eprint(f"[warn] {repoA} vs {repoB} layer={layer} cat={cat}: {e}")

    return rows

# ============ MAIN ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_with_control")
    ap.add_argument("--max-layers", type=int, default=46)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    ensure_outdir(args.outdir)
    revision = "main"

    print("=" * 60)
    print("SOLAR-OPEN-100B vs GLM-4.5-AIR PROVENANCE ANALYSIS")
    print("With CONTROL model: Qwen2-57B-A14B-Instruct")
    print("=" * 60)

    # ============ TOKENIZER EXPANSION ANALYSIS ============
    print("\n[1/3] Analyzing tokenizer expansion...")

    tok_analysis = analyze_tokenizer_expansion(MODEL_A, MODEL_B, revision, token)
    write_json(os.path.join(args.outdir, "tokenizer_expansion_analysis.json"), tok_analysis)

    print(f"  Solar vocab: {tok_analysis['vocab_size_A']}")
    print(f"  GLM vocab:   {tok_analysis['vocab_size_B']}")
    print(f"  Vocab diff:  {tok_analysis['vocab_diff']} (Solar has {tok_analysis['vocab_diff']} more tokens)")
    print(f"  GLM tokens found in Solar: {tok_analysis['b_tokens_found_in_a']} ({tok_analysis['b_tokens_found_in_a_ratio']*100:.1f}%)")
    print(f"  First 1000 merges exact match: {tok_analysis['first_1000_merges_exact_match']} ({tok_analysis['first_1000_merges_exact_match_ratio']*100:.1f}%)")
    print(f"  GLM merges found in Solar: {tok_analysis['merges_b_in_a']} ({tok_analysis['merges_b_in_a_ratio']*100:.1f}%)")

    # ============ WEIGHT COMPARISONS ============
    print("\n[2/3] Comparing weights: Solar vs GLM (MAIN)...")
    rows_main = compare_models_layerwise(MODEL_A, MODEL_B, revision, token, args.max_layers)
    print(f"  Got {len(rows_main)} comparison rows")

    print("\n[3/3] Comparing weights: GLM vs Qwen2-MoE (CONTROL)...")
    rows_control = compare_models_layerwise(MODEL_B, MODEL_CONTROL, revision, token, args.max_layers)
    print(f"  Got {len(rows_control)} comparison rows")

    # ============ AGGREGATE BY LAYER ============
    def aggregate_by_layer(rows: List[Dict], category: str) -> Tuple[List[int], List[float]]:
        per_layer = defaultdict(list)
        for r in rows:
            if r["category"] == category:
                per_layer[r["layer"]].append(r["cosine"])

        layers = sorted(per_layer.keys())
        means = [np.mean(per_layer[L]) for L in layers]
        return layers, means

    # ============ PLOTTING ============
    print("\n[4/4] Generating comparison plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: norm_pre comparison
    ax1 = axes[0]

    layers_main, cos_main = aggregate_by_layer(rows_main, "norm_pre")
    layers_ctrl, cos_ctrl = aggregate_by_layer(rows_control, "norm_pre")

    if layers_main and cos_main:
        ax1.plot(layers_main, cos_main, 'b-o', label='Solar vs GLM (MAIN)', linewidth=2, markersize=6)
    if layers_ctrl and cos_ctrl:
        ax1.plot(layers_ctrl, cos_ctrl, 'r--s', label='GLM vs Qwen2 (CONTROL)', linewidth=2, markersize=5, alpha=0.7)

    ax1.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('input_layernorm (norm_pre)\nSolar-GLM vs Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.05)

    # Plot 2: norm_post comparison
    ax2 = axes[1]

    layers_main2, cos_main2 = aggregate_by_layer(rows_main, "norm_post")
    layers_ctrl2, cos_ctrl2 = aggregate_by_layer(rows_control, "norm_post")

    if layers_main2 and cos_main2:
        ax2.plot(layers_main2, cos_main2, 'b-o', label='Solar vs GLM (MAIN)', linewidth=2, markersize=6)
    if layers_ctrl2 and cos_ctrl2:
        ax2.plot(layers_ctrl2, cos_ctrl2, 'r--s', label='GLM vs Qwen2 (CONTROL)', linewidth=2, markersize=5, alpha=0.7)

    ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('post_attention_layernorm (norm_post)\nSolar-GLM vs Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "comparison_with_control.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    # ============ STATISTICS ============
    def calc_stats(rows: List[Dict], category: str) -> Dict[str, float]:
        vals = [r["cosine"] for r in rows if r["category"] == category and not np.isnan(r["cosine"])]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    stats = {
        "Solar_vs_GLM": {
            "norm_pre": calc_stats(rows_main, "norm_pre"),
            "norm_post": calc_stats(rows_main, "norm_post"),
            "router_or_gate": calc_stats(rows_main, "router_or_gate"),
        },
        "GLM_vs_Qwen2_CONTROL": {
            "norm_pre": calc_stats(rows_control, "norm_pre"),
            "norm_post": calc_stats(rows_control, "norm_post"),
            "router_or_gate": calc_stats(rows_control, "router_or_gate"),
        },
    }

    write_json(os.path.join(args.outdir, "statistics.json"), stats)

    # ============ FINAL REPORT ============
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Cosine Similarity Statistics ---")
    print(f"{'Comparison':<30} {'norm_pre':<20} {'norm_post':<20}")
    print("-" * 70)

    for name, s in stats.items():
        np_mean = s["norm_pre"]["mean"]
        npo_mean = s["norm_post"]["mean"]
        np_str = f"{np_mean:.4f}" if np_mean else "N/A"
        npo_str = f"{npo_mean:.4f}" if npo_mean else "N/A"
        print(f"{name:<30} {np_str:<20} {npo_str:<20}")

    print("\n--- INTERPRETATION ---")

    main_np = stats["Solar_vs_GLM"]["norm_pre"]["mean"] or 0
    ctrl_np = stats["GLM_vs_Qwen2_CONTROL"]["norm_pre"]["mean"] or 0
    main_npo = stats["Solar_vs_GLM"]["norm_post"]["mean"] or 0
    ctrl_npo = stats["GLM_vs_Qwen2_CONTROL"]["norm_post"]["mean"] or 0

    if main_np > 0.8 and ctrl_np < 0.3:
        print(f"[STRONG EVIDENCE] norm_pre: Solar-GLM ({main_np:.3f}) >> Control ({ctrl_np:.3f})")
    elif main_np > ctrl_np + 0.2:
        print(f"[MODERATE EVIDENCE] norm_pre: Solar-GLM ({main_np:.3f}) > Control ({ctrl_np:.3f})")
    else:
        print(f"[WEAK/NO EVIDENCE] norm_pre: Solar-GLM ({main_np:.3f}) vs Control ({ctrl_np:.3f})")

    if main_npo > 0.8 and ctrl_npo < 0.3:
        print(f"[STRONG EVIDENCE] norm_post: Solar-GLM ({main_npo:.3f}) >> Control ({ctrl_npo:.3f})")
    elif main_npo > ctrl_npo + 0.2:
        print(f"[MODERATE EVIDENCE] norm_post: Solar-GLM ({main_npo:.3f}) > Control ({ctrl_npo:.3f})")
    else:
        print(f"[WEAK/NO EVIDENCE] norm_post: Solar-GLM ({main_npo:.3f}) vs Control ({ctrl_npo:.3f})")

    # Tokenizer expansion interpretation
    print("\n--- TOKENIZER EXPANSION ANALYSIS ---")
    if tok_analysis["vocab_diff"] > 10000:
        print(f"[INFO] Solar has {tok_analysis['vocab_diff']} MORE tokens than GLM")
        print(f"       This is consistent with tokenizer EXPANSION (adding tokens to GLM's tokenizer)")

    if tok_analysis["b_tokens_found_in_a_ratio"] > 0.9:
        print(f"[STRONG EVIDENCE] {tok_analysis['b_tokens_found_in_a_ratio']*100:.1f}% of GLM tokens exist in Solar")
    elif tok_analysis["b_tokens_found_in_a_ratio"] > 0.5:
        print(f"[MODERATE EVIDENCE] {tok_analysis['b_tokens_found_in_a_ratio']*100:.1f}% of GLM tokens exist in Solar")

    if tok_analysis["first_1000_merges_exact_match_ratio"] > 0.5:
        print(f"[EVIDENCE] First 1000 BPE merges: {tok_analysis['first_1000_merges_exact_match_ratio']*100:.1f}% match")

    print("\n" + "=" * 60)
    print(f"Full results saved to: {args.outdir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
