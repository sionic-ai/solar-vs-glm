#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_final.py

Final comprehensive analysis with CORRECT baseline methodology.

Key insight: LayerNorm weights ≈ 1.0 + small_noise, so raw cosine is always high.
We need to compare:
1. DEVIATION patterns (subtract mean of 1.0)
2. Or use a proper control: compare unrelated layers within same model
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url

MODEL_A = "upstage/Solar-Open-100B"
MODEL_B = "zai-org/GLM-4.5-Air"

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
        raise RuntimeError(f"Header too large")
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
    else:
        arr = np.frombuffer(raw, dtype=np.float32)
    size = int(np.prod(shape)) if shape else 1
    return arr[:size].reshape(shape)

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
    return int(info["data_offsets"][1]) - int(info["data_offsets"][0])

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64) - np.mean(a)
    b = b.reshape(-1).astype(np.float64) - np.mean(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of (a - 1) vs (b - 1) to capture deviation patterns."""
    a = a.reshape(-1).astype(np.float64) - 1.0
    b = b.reshape(-1).astype(np.float64) - 1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def inv_vocab(tok_json: Dict[str, Any]) -> List[str]:
    vocab = tok_json["model"]["vocab"]
    inv = [None] * len(vocab)
    for t, i in vocab.items():
        inv[i] = t
    return inv

def deep_tokenizer_analysis(repoA: str, repoB: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    tjA = load_tokenizer_json(repoA, revision, token)
    tjB = load_tokenizer_json(repoB, revision, token)

    vocabA = tjA["model"]["vocab"]
    vocabB = tjB["model"]["vocab"]

    common_tokens = set(vocabA.keys()) & set(vocabB.keys())
    shifts = [vocabA[tok] - vocabB[tok] for tok in common_tokens]
    shift_counter = Counter(shifts)
    most_common_shifts = shift_counter.most_common(20)

    mergesA = tjA["model"].get("merges", []) or []
    mergesB = tjB["model"].get("merges", []) or []

    def norm_merge(m):
        return m if isinstance(m, str) else " ".join(map(str, m))

    setA = set(norm_merge(m) for m in mergesA)
    setB = set(norm_merge(m) for m in mergesB)

    return {
        "vocab_A": len(vocabA),
        "vocab_B": len(vocabB),
        "vocab_diff": len(vocabA) - len(vocabB),
        "common_tokens": len(common_tokens),
        "common_ratio": len(common_tokens) / len(vocabB) if vocabB else 0,
        "top_shifts": [{"shift": s, "count": c} for s, c in most_common_shifts[:10]],
        "merges_A": len(mergesA),
        "merges_B": len(mergesB),
        "merges_common": len(setA & setB),
        "merges_B_in_A_ratio": len(setA & setB) / len(setB) if setB else 0,
    }

LAYER_RE = re.compile(r"layers\.(\d+)\.")

def get_layer_index(key: str) -> Optional[int]:
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None

def classify_key(key: str) -> Optional[str]:
    lk = key.lower()
    if not lk.endswith(".weight"):
        return None
    if any(x in lk for x in ["router", "gate", "gating"]) and "proj" not in lk:
        return "router"
    if "norm" in lk or "rms" in lk:
        if any(x in lk for x in ["input", "pre", "attn_norm", "ln_1"]):
            return "norm_pre"
        if any(x in lk for x in ["post", "ffn", "mlp", "ln_2"]):
            return "norm_post"
        return "norm_any"
    return None

def build_keyrefs(weight_map: Dict[str, str], max_layers: int) -> Dict[int, Dict[str, Any]]:
    per_layer = defaultdict(lambda: defaultdict(list))
    for k, shard in weight_map.items():
        li = get_layer_index(k)
        if li is None or li >= max_layers:
            continue
        cat = classify_key(k)
        if cat:
            per_layer[li][cat].append({"key": k, "shard": shard})

    out = {}
    for li, cats in per_layer.items():
        chosen = {}
        for cat, lst in cats.items():
            chosen[cat] = sorted(lst, key=lambda x: x["key"])[0]
        if "norm_pre" not in chosen and "norm_any" in chosen:
            chosen["norm_pre"] = chosen["norm_any"]
        if "norm_post" not in chosen and "norm_any" in chosen:
            chosen["norm_post"] = chosen["norm_any"]
        out[li] = chosen
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_final")
    ap.add_argument("--max-layers", type=int, default=46)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    ensure_outdir(args.outdir)
    revision = "main"
    max_header_bytes = 64 * 1024 * 1024
    max_tensor_bytes = 2_500_000

    print("=" * 70)
    print("SOLAR-OPEN-100B vs GLM-4.5-AIR: PROVENANCE PROOF")
    print("=" * 70)

    # Tokenizer analysis
    print("\n[1/3] Tokenizer analysis...")
    tok = deep_tokenizer_analysis(MODEL_A, MODEL_B, revision, token)
    write_json(os.path.join(args.outdir, "tokenizer.json"), tok)
    print(f"  Solar: {tok['vocab_A']:,} tokens, GLM: {tok['vocab_B']:,} tokens")
    print(f"  Diff: +{tok['vocab_diff']:,} (Solar has MORE)")
    print(f"  Common: {tok['common_tokens']:,} ({tok['common_ratio']*100:.1f}% of GLM)")
    print(f"  Merges: {tok['merges_B_in_A_ratio']*100:.1f}% of GLM merges in Solar")

    # Load models
    print("\n[2/3] Loading model indices...")
    cfgA = load_config(MODEL_A, revision, token)
    cfgB = load_config(MODEL_B, revision, token)
    idxA = load_index_json(MODEL_A, revision, token)
    idxB = load_index_json(MODEL_B, revision, token)

    wmA, wmB = idxA["weight_map"], idxB["weight_map"]
    layersA = cfgA.get("num_hidden_layers", 0)
    layersB = cfgB.get("num_hidden_layers", 0)
    probe_layers = min(layersA, layersB, args.max_layers)

    refsA = build_keyrefs(wmA, layersA)
    refsB = build_keyrefs(wmB, layersB)
    common_layers = sorted(set(refsA) & set(refsB))[:probe_layers]

    print(f"  Solar: {layersA} layers, GLM: {layersB} layers")
    print(f"  Comparing {len(common_layers)} layers")

    header_cache = {}
    def get_header(repo, shard):
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_safetensors_header(repo, shard, revision, token, max_header_bytes)
        return header_cache[k]

    # Collect all LayerNorm weights
    print("\n[3/3] Comparing weights...")
    solar_norms = []  # List of (layer, cat, tensor)
    glm_norms = []

    for layer in common_layers:
        for cat in ["norm_pre", "norm_post"]:
            if cat not in refsA.get(layer, {}) or cat not in refsB.get(layer, {}):
                continue

            refA, refB = refsA[layer][cat], refsB[layer][cat]
            try:
                hA = get_header(MODEL_A, refA["shard"])
                hB = get_header(MODEL_B, refB["shard"])

                if refA["key"] not in hA or refB["key"] not in hB:
                    continue
                if tensor_nbytes(hA[refA["key"]]) > max_tensor_bytes:
                    continue

                A = fetch_tensor(MODEL_A, refA["shard"], revision, refA["key"], hA, token)
                B = fetch_tensor(MODEL_B, refB["shard"], revision, refB["key"], hB, token)

                if A.shape != B.shape:
                    continue

                solar_norms.append((layer, cat, A))
                glm_norms.append((layer, cat, B))
            except Exception as e:
                eprint(f"[warn] {layer}/{cat}: {e}")

    print(f"  Collected {len(solar_norms)} norm pairs")

    # Compute similarities
    rows = []
    for (layerA, catA, A), (layerB, catB, B) in zip(solar_norms, glm_norms):
        rows.append({
            "layer": layerA,
            "cat": catA,
            "cosine": cosine(A, B),
            "pearson": pearson(A, B),
            "centered_cosine": centered_cosine(A, B),
        })

    # CONTROL: Compare mismatched layers within same model pair
    # This shows what similarity looks like for UNRELATED layers
    control_cosines = []
    control_centered = []

    for i in range(min(30, len(solar_norms))):
        for j in range(i + 5, min(i + 10, len(glm_norms))):  # Compare layer i with layer j
            A = solar_norms[i][2]
            B = glm_norms[j][2]
            if A.shape == B.shape:
                control_cosines.append(cosine(A, B))
                control_centered.append(centered_cosine(A, B))

    # Statistics
    matched_cosines = [r["cosine"] for r in rows if not np.isnan(r["cosine"])]
    matched_centered = [r["centered_cosine"] for r in rows if not np.isnan(r["centered_cosine"])]
    matched_pearsons = [r["pearson"] for r in rows if not np.isnan(r["pearson"])]

    stats = {
        "matched_layers": {
            "cosine_mean": float(np.mean(matched_cosines)),
            "cosine_std": float(np.std(matched_cosines)),
            "centered_cosine_mean": float(np.mean(matched_centered)),
            "centered_cosine_std": float(np.std(matched_centered)),
            "pearson_mean": float(np.mean(matched_pearsons)),
        },
        "control_mismatched": {
            "cosine_mean": float(np.mean(control_cosines)) if control_cosines else None,
            "centered_cosine_mean": float(np.mean(control_centered)) if control_centered else None,
        }
    }
    write_json(os.path.join(args.outdir, "statistics.json"), stats)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw cosine by layer
    ax1 = axes[0, 0]
    for cat, color in [("norm_pre", "blue"), ("norm_post", "green")]:
        xs = [r["layer"] for r in rows if r["cat"] == cat]
        ys = [r["cosine"] for r in rows if r["cat"] == cat]
        ax1.plot(xs, ys, 'o-', color=color, label=cat, markersize=5)

    if control_cosines:
        ax1.axhline(np.mean(control_cosines), color='red', linestyle='--',
                   label=f'Mismatched layers ({np.mean(control_cosines):.4f})')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Raw Cosine: Matched Layers (Solar[i] vs GLM[i])')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.01)

    # Plot 2: Centered cosine (deviation patterns)
    ax2 = axes[0, 1]
    for cat, color in [("norm_pre", "blue"), ("norm_post", "green")]:
        xs = [r["layer"] for r in rows if r["cat"] == cat]
        ys = [r["centered_cosine"] for r in rows if r["cat"] == cat]
        ax2.plot(xs, ys, 'o-', color=color, label=cat, markersize=5)

    if control_centered:
        ax2.axhline(np.mean(control_centered), color='red', linestyle='--',
                   label=f'Mismatched layers ({np.mean(control_centered):.4f})')
    ax2.axhline(0, color='gray', linestyle=':')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Centered Cosine (deviation from 1.0)')
    ax2.set_title('Centered Cosine: Do DEVIATIONS from 1.0 match?')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histogram comparison
    ax3 = axes[1, 0]
    ax3.hist(matched_centered, bins=30, alpha=0.7, label='Matched layers (Solar[i] vs GLM[i])', color='blue')
    if control_centered:
        ax3.hist(control_centered, bins=30, alpha=0.5, label='Mismatched layers (control)', color='red')
    ax3.axvline(np.mean(matched_centered), color='blue', linestyle='--', linewidth=2)
    if control_centered:
        ax3.axvline(np.mean(control_centered), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Centered Cosine')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution: Matched vs Mismatched Layer Pairs')
    ax3.legend()

    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    ctrl_cos = stats["control_mismatched"]["cosine_mean"] or 0
    ctrl_cent = stats["control_mismatched"]["centered_cosine_mean"] or 0

    summary = f"""
PROVENANCE EVIDENCE SUMMARY
===========================

1. RAW COSINE SIMILARITY (LayerNorm weights)
   Matched layers (Solar[i] vs GLM[i]):  {stats['matched_layers']['cosine_mean']:.4f}
   Mismatched layers (control):          {ctrl_cos:.4f}

   -> Both high because LayerNorms ≈ 1.0 (uninformative)

2. CENTERED COSINE (deviation patterns)
   Matched layers:    {stats['matched_layers']['centered_cosine_mean']:.4f}
   Mismatched layers: {ctrl_cent:.4f}

   -> THIS IS THE KEY METRIC!
   -> If centered cosine >> 0, the DEVIATION PATTERNS match
   -> Random/unrelated models: centered cosine ≈ 0

3. TOKENIZER ANALYSIS
   Solar = GLM + {tok['vocab_diff']:,} new tokens
   {tok['common_ratio']*100:.1f}% of GLM tokens found in Solar
   {tok['merges_B_in_A_ratio']*100:.1f}% of GLM BPE merges preserved

4. CONCLUSION
"""

    if stats['matched_layers']['centered_cosine_mean'] > 0.3 and ctrl_cent < 0.1:
        conclusion = """   [STRONG EVIDENCE] Solar derived from GLM
   - Deviation patterns MATCH (centered cosine >> baseline)
   - Tokenizer shows EXPANSION pattern
   - Consistent with CONTINUAL PRETRAINING"""
    elif stats['matched_layers']['centered_cosine_mean'] > 0.1:
        conclusion = """   [MODERATE EVIDENCE] Possible shared lineage
   - Some deviation pattern matching detected"""
    else:
        conclusion = """   [WEAK EVIDENCE] Inconclusive"""

    ax4.text(0.02, 0.98, summary + conclusion, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "proof_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Print final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nMatched layers (Solar[i] vs GLM[i]):")
    print(f"  Raw cosine:      {stats['matched_layers']['cosine_mean']:.4f}")
    print(f"  Centered cosine: {stats['matched_layers']['centered_cosine_mean']:.4f}")
    print(f"  Pearson:         {stats['matched_layers']['pearson_mean']:.4f}")

    print(f"\nMismatched layers (control):")
    print(f"  Raw cosine:      {ctrl_cos:.4f}")
    print(f"  Centered cosine: {ctrl_cent:.4f}")

    diff = stats['matched_layers']['centered_cosine_mean'] - ctrl_cent
    print(f"\nDIFFERENCE (matched - control): {diff:.4f}")

    if diff > 0.2:
        print("\n[VERDICT] STRONG EVIDENCE of shared origin")
    elif diff > 0.05:
        print("\n[VERDICT] MODERATE EVIDENCE of shared origin")
    else:
        print("\n[VERDICT] WEAK/NO EVIDENCE")

    print(f"\nPlot saved: {plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
