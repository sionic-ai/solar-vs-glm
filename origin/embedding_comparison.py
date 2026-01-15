#!/usr/bin/env python3
"""
embedding_comparison.py

Compare embedding vectors for common English tokens between Solar and GLM.

Key test: If Solar is derived from GLM, embeddings for common tokens should be similar.
"""

import json
import os
import re
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url

SOLAR = "upstage/Solar-Open-100B"
GLM = "zai-org/GLM-4.5-Air"

def http_range_get(url: str, start: int, end: int, token: Optional[str], retries: int = 5) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=300, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}", file=sys.stderr)
        time.sleep(1 * (attempt + 1))
    raise RuntimeError("Range GET failed")

def load_tokenizer(repo: str, rev: str, token: Optional[str]) -> Dict[str, int]:
    """Load tokenizer and return vocab dict (token -> id)"""
    path = hf_hub_download(repo, "tokenizer.json", revision=rev, token=token)
    with open(path) as f:
        tj = json.load(f)
    return tj["model"]["vocab"]

def load_index(repo: str, rev: str, token: Optional[str]) -> Dict[str, str]:
    path = hf_hub_download(repo, "model.safetensors.index.json", revision=rev, token=token)
    with open(path) as f:
        return json.load(f)["weight_map"]

def fetch_header(repo: str, shard: str, rev: str, token: Optional[str]) -> Dict:
    url = hf_hub_url(repo_id=repo, filename=shard, revision=rev)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = struct.unpack("<Q", raw8)[0]
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header

def fetch_embedding_row(url: str, header: Dict, key: str, row_idx: int, hidden_size: int, token: Optional[str]) -> np.ndarray:
    """Fetch a single row from embedding matrix"""
    info = header[key]
    dtype = info["dtype"]

    # BF16 = 2 bytes per element
    elem_size = 2 if dtype == "BF16" else 4
    row_bytes = hidden_size * elem_size

    off0 = info["data_offsets"][0]
    row_start = 8 + header["__header_len__"] + off0 + row_idx * row_bytes
    row_end = row_start + row_bytes - 1

    raw = http_range_get(url, row_start, row_end, token)

    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    else:
        return np.frombuffer(raw, dtype=np.float32)

def is_english_token(token: str) -> bool:
    """Check if token is primarily English (ASCII printable)"""
    # Remove BPE markers
    clean = token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
    # Check if mostly ASCII
    if not clean:
        return False
    ascii_chars = sum(1 for c in clean if 32 <= ord(c) <= 126)
    return ascii_chars >= len(clean) * 0.8 and len(clean) >= 2

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def main():
    token = os.environ.get("HF_TOKEN")
    rev = "main"
    hidden_size = 4096  # Both models have hidden_size=4096

    print("=" * 70)
    print("EMBEDDING COMPARISON: Solar vs GLM (English Tokens)")
    print("=" * 70)

    # Load tokenizers
    print("\n[1/4] Loading tokenizers...")
    solar_vocab = load_tokenizer(SOLAR, rev, token)
    glm_vocab = load_tokenizer(GLM, rev, token)

    print(f"  Solar vocab size: {len(solar_vocab)}")
    print(f"  GLM vocab size: {len(glm_vocab)}")

    # Find common English tokens
    print("\n[2/4] Finding common English tokens...")
    common_tokens = set(solar_vocab.keys()) & set(glm_vocab.keys())
    english_common = [t for t in common_tokens if is_english_token(t)]

    # Sort by frequency (lower ID = more common)
    english_common.sort(key=lambda t: min(solar_vocab[t], glm_vocab[t]))

    print(f"  Common tokens: {len(common_tokens)}")
    print(f"  English common: {len(english_common)}")
    print(f"  Sample tokens: {english_common[:20]}")

    # Load weight map and headers
    print("\n[3/4] Loading embedding weight info...")
    solar_wm = load_index(SOLAR, rev, token)
    glm_wm = load_index(GLM, rev, token)

    # Find embedding key
    embed_key = "model.embed_tokens.weight"
    solar_shard = solar_wm[embed_key]
    glm_shard = glm_wm[embed_key]

    solar_hdr = fetch_header(SOLAR, solar_shard, rev, token)
    glm_hdr = fetch_header(GLM, glm_shard, rev, token)

    solar_shape = solar_hdr[embed_key]["shape"]
    glm_shape = glm_hdr[embed_key]["shape"]

    print(f"  Solar embed shape: {solar_shape}")
    print(f"  GLM embed shape: {glm_shape}")

    # Sample tokens for comparison
    print("\n[4/4] Comparing embeddings for English tokens...")

    # Take first 200 common English tokens
    sample_tokens = english_common[:200]

    results = []

    for i, tok in enumerate(sample_tokens):
        solar_id = solar_vocab[tok]
        glm_id = glm_vocab[tok]

        try:
            # Fetch embedding rows
            solar_emb = fetch_embedding_row(
                solar_hdr["__url__"], solar_hdr, embed_key, solar_id, hidden_size, token
            )
            glm_emb = fetch_embedding_row(
                glm_hdr["__url__"], glm_hdr, embed_key, glm_id, hidden_size, token
            )

            cos = cosine(solar_emb, glm_emb)
            results.append({
                "token": tok,
                "solar_id": solar_id,
                "glm_id": glm_id,
                "cosine": cos,
            })

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(sample_tokens)} tokens...")

        except Exception as e:
            print(f"  [skip] {tok}: {e}", file=sys.stderr)

    # Compute statistics
    cosines = [r["cosine"] for r in results if not np.isnan(r["cosine"])]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nCompared {len(results)} English tokens")
    print(f"\nEmbedding Cosine Similarity:")
    print(f"  Mean:   {np.mean(cosines):.4f}")
    print(f"  Std:    {np.std(cosines):.4f}")
    print(f"  Min:    {np.min(cosines):.4f}")
    print(f"  Max:    {np.max(cosines):.4f}")
    print(f"  Median: {np.median(cosines):.4f}")

    # High similarity tokens
    high_sim = [r for r in results if r["cosine"] > 0.9]
    print(f"\nTokens with cosine > 0.9: {len(high_sim)}/{len(results)} ({100*len(high_sim)/len(results):.1f}%)")

    # Low similarity tokens
    low_sim = [r for r in results if r["cosine"] < 0.1]
    print(f"Tokens with cosine < 0.1: {len(low_sim)}/{len(results)} ({100*len(low_sim)/len(results):.1f}%)")

    # Within-model baseline: Compare random pairs within GLM
    print("\n[BASELINE] Comparing random token pairs within GLM...")
    baseline_cosines = []

    glm_tokens = list(glm_vocab.items())
    np.random.seed(42)

    for _ in range(50):
        tok1, id1 = glm_tokens[np.random.randint(1000, 10000)]
        tok2, id2 = glm_tokens[np.random.randint(1000, 10000)]

        if id1 == id2:
            continue

        try:
            emb1 = fetch_embedding_row(
                glm_hdr["__url__"], glm_hdr, embed_key, id1, hidden_size, token
            )
            emb2 = fetch_embedding_row(
                glm_hdr["__url__"], glm_hdr, embed_key, id2, hidden_size, token
            )
            baseline_cosines.append(cosine(emb1, emb2))
        except:
            pass

    if baseline_cosines:
        print(f"\nBaseline (random GLM pairs): mean={np.mean(baseline_cosines):.4f}, std={np.std(baseline_cosines):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(cosines, bins=50, alpha=0.7, color='blue', label='Solar-GLM (same token)', edgecolor='black')
    if baseline_cosines:
        ax1.hist(baseline_cosines, bins=30, alpha=0.5, color='gray', label='Baseline (random pairs)')
    ax1.axvline(np.mean(cosines), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cosines):.3f}')
    if baseline_cosines:
        ax1.axvline(np.mean(baseline_cosines), color='gray', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Embedding Similarity: Common English Tokens\nSolar vs GLM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter by token ID
    ax2 = axes[1]
    ids = [r["glm_id"] for r in results]
    cos_vals = [r["cosine"] for r in results]
    ax2.scatter(ids, cos_vals, alpha=0.5, s=10)
    ax2.axhline(np.mean(cosines), color='red', linestyle='--', label=f'Mean: {np.mean(cosines):.3f}')
    ax2.set_xlabel('GLM Token ID')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Embedding Similarity by Token ID')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/embedding_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved: results/embedding_comparison.png")

    # Final interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    if np.mean(cosines) > 0.5:
        print("\n[STRONG EVIDENCE] Embedding mean cosine > 0.5")
        print("Common English tokens have SIMILAR embeddings")
        print("This supports the DERIVATION hypothesis")
    elif np.mean(cosines) > 0.2:
        print("\n[MODERATE EVIDENCE] Embedding mean cosine > 0.2")
        print("Some similarity in embeddings detected")
    else:
        print("\n[WEAK EVIDENCE] Embedding mean cosine < 0.2")
        print("Embeddings appear independent")

    if baseline_cosines:
        diff = np.mean(cosines) - np.mean(baseline_cosines)
        print(f"\nDifference from baseline: {diff:.4f}")
        if diff > 0.1:
            print("Cross-model similarity >> within-model baseline")
            print("This is additional evidence of DERIVATION")

    # Save detailed results
    with open('results/embedding_comparison.json', 'w') as f:
        json.dump({
            "num_tokens": len(results),
            "mean_cosine": float(np.mean(cosines)),
            "std_cosine": float(np.std(cosines)),
            "min_cosine": float(np.min(cosines)),
            "max_cosine": float(np.max(cosines)),
            "baseline_mean": float(np.mean(baseline_cosines)) if baseline_cosines else None,
            "high_similarity_count": len(high_sim),
            "low_similarity_count": len(low_sim),
            "sample_results": results[:50],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: results/embedding_comparison.json")

if __name__ == "__main__":
    main()
