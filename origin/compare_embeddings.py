#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare tokenizer embeddings between Solar-Open-100B and GLM-4.5-Air.

This script:
1. Downloads tokenizer vocabularies from both models
2. Finds common tokens (especially English/ASCII tokens)
3. Samples embedding weights via HTTP Range requests
4. Calculates cosine similarities to verify derivation hypothesis

Key insight: If Solar derived from GLM, common tokens should have similar embeddings.
English tokens are unlikely to be part of the 45K new tokens (probably Korean).
"""

import argparse
import json
import os
import re
import struct
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import requests

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4}


def hf_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _headers(token: Optional[str], extra: Optional[dict] = None) -> dict:
    h = {"Accept-Encoding": "identity"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update(extra)
    return h


def http_get(url: str, token: Optional[str]) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=_headers(token), allow_redirects=True, timeout=120)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        print(f"[WARN] http_get failed: {e}")
    return None


def http_get_json(url: str, token: Optional[str]) -> Optional[dict]:
    b = http_get(url, token)
    if b is None:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def http_range_get(url: str, start: int, end: int, token: Optional[str]) -> bytes:
    r = requests.get(
        url,
        headers=_headers(token, {"Range": f"bytes={start}-{end}"}),
        allow_redirects=True,
        timeout=120,
    )
    if r.status_code == 206:
        return r.content
    expected = end - start + 1
    if r.status_code == 200 and len(r.content) == expected:
        return r.content
    raise RuntimeError(f"Range GET failed: HTTP {r.status_code}")


@dataclass
class TensorInfo:
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]


@dataclass
class SafetensorsHeader:
    header_len: int
    base_data_offset: int
    tensors: Dict[str, TensorInfo]


def parse_safetensors_header(url: str, token: Optional[str]) -> SafetensorsHeader:
    b0 = http_range_get(url, 0, 7, token)
    header_len = struct.unpack("<Q", b0)[0]
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header_json = json.loads(hb.decode("utf-8"))

    tensors = {}
    for k, v in header_json.items():
        if k == "__metadata__":
            continue
        tensors[k] = TensorInfo(
            dtype=v["dtype"],
            shape=tuple(int(x) for x in v["shape"]),
            data_offsets=(int(v["data_offsets"][0]), int(v["data_offsets"][1])),
        )

    return SafetensorsHeader(
        header_len=int(header_len),
        base_data_offset=8 + int(header_len),
        tensors=tensors,
    )


def bytes_to_f32(buf: bytes, dtype: str) -> np.ndarray:
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32)
    if dtype == "F16":
        return np.frombuffer(buf, dtype=np.float16).astype(np.float32)
    if dtype == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    nx = np.linalg.norm(x64)
    ny = np.linalg.norm(y64)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x64, y64) / (nx * ny))


def load_tokenizer_vocab(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, int]:
    """Load tokenizer vocabulary from HuggingFace."""
    vocab = {}

    # Try tokenizer.json first (most common)
    url = hf_url(repo_id, revision, "tokenizer.json")
    data = http_get_json(url, token)
    if data and "model" in data and "vocab" in data["model"]:
        vocab = data["model"]["vocab"]
        print(f"  Loaded from tokenizer.json: {len(vocab)} tokens")
        return vocab

    # Try vocab.json
    url = hf_url(repo_id, revision, "vocab.json")
    data = http_get_json(url, token)
    if data:
        vocab = data
        print(f"  Loaded from vocab.json: {len(vocab)} tokens")
        return vocab

    # Try tokenizer_config.json for sentencepiece
    url = hf_url(repo_id, revision, "tokenizer_config.json")
    data = http_get_json(url, token)
    if data:
        print(f"  Found tokenizer_config.json, tokenizer type may be sentencepiece")

    return vocab


def is_english_token(token: str) -> bool:
    """Check if token is primarily English/ASCII."""
    # Remove special prefixes like Ġ (GPT-2 style) or ▁ (sentencepiece)
    clean = token.replace("Ġ", "").replace("▁", "").replace("Ã", "").replace("Â", "")

    if not clean:
        return False

    # Check if mostly ASCII letters/common punctuation
    ascii_chars = sum(1 for c in clean if ord(c) < 128)
    return ascii_chars / len(clean) > 0.8 and len(clean) >= 2


def is_pure_ascii_word(token: str) -> bool:
    """Check if token is a pure ASCII word (letters only)."""
    clean = token.replace("Ġ", "").replace("▁", "").strip()
    return clean.isascii() and clean.isalpha() and len(clean) >= 3


def find_common_tokens(vocab_a: Dict[str, int], vocab_b: Dict[str, int]) -> List[Tuple[str, int, int]]:
    """Find tokens that exist in both vocabularies."""
    common = []
    for token, id_a in vocab_a.items():
        if token in vocab_b:
            id_b = vocab_b[token]
            common.append((token, id_a, id_b))
    return common


class EmbeddingComparer:
    def __init__(self, repo_id: str, revision: str, token: Optional[str]):
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.name = repo_id.split("/")[-1]

        # Find embedding tensor
        self.embed_url, self.embed_info, self.header = self._find_embedding_tensor()

    def _find_embedding_tensor(self) -> Tuple[str, TensorInfo, SafetensorsHeader]:
        """Find the embedding tensor in safetensors files."""
        # Get weight map
        index_url = hf_url(self.repo_id, self.revision, "model.safetensors.index.json")
        index = http_get_json(index_url, self.token)

        if not index or "weight_map" not in index:
            raise RuntimeError(f"Could not load index for {self.repo_id}")

        weight_map = index["weight_map"]

        # Find embedding key
        embed_keys = [
            "model.embed_tokens.weight",
            "transformer.word_embeddings.weight",
            "transformer.wte.weight",
            "model.embeddings.word_embeddings.weight",
        ]

        embed_key = None
        for k in embed_keys:
            if k in weight_map:
                embed_key = k
                break

        if not embed_key:
            # Search for any embedding-like key
            for k in weight_map.keys():
                if "embed" in k.lower() and "weight" in k.lower():
                    embed_key = k
                    break

        if not embed_key:
            raise RuntimeError(f"Could not find embedding tensor in {self.repo_id}")

        shard_file = weight_map[embed_key]
        shard_url = hf_url(self.repo_id, self.revision, shard_file)
        header = parse_safetensors_header(shard_url, self.token)

        if embed_key not in header.tensors:
            raise RuntimeError(f"Embedding key {embed_key} not in shard header")

        print(f"  {self.name}: Found {embed_key} in {shard_file}")
        print(f"    Shape: {header.tensors[embed_key].shape}, Dtype: {header.tensors[embed_key].dtype}")

        return shard_url, header.tensors[embed_key], header

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding vector for a specific token ID."""
        vocab_size, hidden_size = self.embed_info.shape

        if token_id >= vocab_size:
            raise ValueError(f"Token ID {token_id} >= vocab_size {vocab_size}")

        dtype = self.embed_info.dtype
        elem_size = DTYPE_SIZES[dtype]

        # Calculate byte offset for this token's embedding
        row_bytes = hidden_size * elem_size
        data_start = self.embed_info.data_offsets[0]

        token_offset = self.header.base_data_offset + data_start + (token_id * row_bytes)
        token_end = token_offset + row_bytes - 1

        buf = http_range_get(self.embed_url, token_offset, token_end, self.token)
        return bytes_to_f32(buf, dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solar", default="upstage/Solar-Open-100B")
    ap.add_argument("--glm", default="zai-org/GLM-4.5-Air")
    ap.add_argument("--rev_solar", default="main")
    ap.add_argument("--rev_glm", default="main")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--max_tokens", type=int, default=500, help="Max common tokens to compare")
    ap.add_argument("--out_dir", default="embedding_results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("TOKENIZER EMBEDDING COMPARISON")
    print("=" * 70)

    # Load tokenizer vocabularies
    print("\n[1] Loading tokenizer vocabularies...")
    print(f"  Solar: {args.solar}")
    solar_vocab = load_tokenizer_vocab(args.solar, args.rev_solar, args.token)

    print(f"  GLM: {args.glm}")
    glm_vocab = load_tokenizer_vocab(args.glm, args.rev_glm, args.token)

    if not solar_vocab or not glm_vocab:
        print("[ERROR] Could not load vocabularies")
        return

    print(f"\n  Solar vocab size: {len(solar_vocab)}")
    print(f"  GLM vocab size: {len(glm_vocab)}")

    # Find common tokens
    print("\n[2] Finding common tokens...")
    common_tokens = find_common_tokens(solar_vocab, glm_vocab)
    print(f"  Total common tokens: {len(common_tokens)}")

    # Filter for English tokens
    english_common = [(t, a, b) for t, a, b in common_tokens if is_english_token(t)]
    print(f"  English common tokens: {len(english_common)}")

    # Filter for pure ASCII words
    ascii_words = [(t, a, b) for t, a, b in common_tokens if is_pure_ascii_word(t)]
    print(f"  Pure ASCII words: {len(ascii_words)}")

    # Analyze token ID distribution
    print("\n[3] Analyzing token ID patterns...")
    same_id = sum(1 for t, a, b in common_tokens if a == b)
    print(f"  Tokens with SAME ID in both: {same_id} ({100*same_id/len(common_tokens):.1f}%)")

    # Check if GLM token IDs map directly
    glm_max_id = max(b for _, _, b in common_tokens)
    solar_ids_for_glm = [(a, b) for _, a, b in common_tokens if b < 151552]
    same_mapping = sum(1 for a, b in solar_ids_for_glm if a == b)
    print(f"  GLM IDs < 151552 with same Solar ID: {same_mapping}/{len(solar_ids_for_glm)}")

    # Initialize embedding comparers
    print("\n[4] Initializing embedding access...")
    try:
        solar_embed = EmbeddingComparer(args.solar, args.rev_solar, args.token)
        glm_embed = EmbeddingComparer(args.glm, args.rev_glm, args.token)
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Compare embeddings for common tokens
    print(f"\n[5] Comparing embeddings for up to {args.max_tokens} tokens...")

    results = []

    # Prioritize pure ASCII words for clear comparison
    tokens_to_compare = ascii_words[:args.max_tokens]
    if len(tokens_to_compare) < args.max_tokens:
        # Add more English tokens if needed
        remaining = args.max_tokens - len(tokens_to_compare)
        additional = [t for t in english_common if t not in tokens_to_compare][:remaining]
        tokens_to_compare.extend(additional)

    print(f"  Comparing {len(tokens_to_compare)} tokens...")

    for i, (token, solar_id, glm_id) in enumerate(tokens_to_compare):
        try:
            solar_emb = solar_embed.get_embedding(solar_id)
            glm_emb = glm_embed.get_embedding(glm_id)

            cos = cosine(solar_emb, glm_emb)

            results.append({
                "token": token,
                "solar_id": solar_id,
                "glm_id": glm_id,
                "same_id": solar_id == glm_id,
                "cosine": cos,
            })

            if (i + 1) % 50 == 0:
                print(f"    Processed {i+1}/{len(tokens_to_compare)}...")

        except Exception as e:
            print(f"    [WARN] Failed for token '{token}': {e}")
            continue

    print(f"  Successfully compared {len(results)} tokens")

    # Analyze results
    print("\n" + "=" * 70)
    print("EMBEDDING SIMILARITY RESULTS")
    print("=" * 70)

    cosines = [r["cosine"] for r in results if not np.isnan(r["cosine"])]
    same_id_cosines = [r["cosine"] for r in results if r["same_id"] and not np.isnan(r["cosine"])]
    diff_id_cosines = [r["cosine"] for r in results if not r["same_id"] and not np.isnan(r["cosine"])]

    print(f"\n[Overall Statistics]")
    print(f"  Total compared: {len(cosines)}")
    print(f"  Mean cosine: {np.mean(cosines):.6f}")
    print(f"  Std cosine: {np.std(cosines):.6f}")
    print(f"  Min cosine: {np.min(cosines):.6f}")
    print(f"  Max cosine: {np.max(cosines):.6f}")

    # Distribution analysis
    high_sim = sum(1 for c in cosines if c > 0.9)
    med_sim = sum(1 for c in cosines if 0.5 <= c <= 0.9)
    low_sim = sum(1 for c in cosines if c < 0.5)

    print(f"\n[Distribution]")
    print(f"  High similarity (>0.9): {high_sim} ({100*high_sim/len(cosines):.1f}%)")
    print(f"  Medium similarity (0.5-0.9): {med_sim} ({100*med_sim/len(cosines):.1f}%)")
    print(f"  Low similarity (<0.5): {low_sim} ({100*low_sim/len(cosines):.1f}%)")

    if same_id_cosines:
        print(f"\n[Same ID tokens]")
        print(f"  Count: {len(same_id_cosines)}")
        print(f"  Mean cosine: {np.mean(same_id_cosines):.6f}")

    if diff_id_cosines:
        print(f"\n[Different ID tokens]")
        print(f"  Count: {len(diff_id_cosines)}")
        print(f"  Mean cosine: {np.mean(diff_id_cosines):.6f}")

    # Show example tokens
    print(f"\n[Sample High-Similarity Tokens]")
    sorted_results = sorted(results, key=lambda x: x["cosine"], reverse=True)
    for r in sorted_results[:10]:
        print(f"  '{r['token']}': cos={r['cosine']:.4f} (solar_id={r['solar_id']}, glm_id={r['glm_id']})")

    print(f"\n[Sample Low-Similarity Tokens]")
    for r in sorted_results[-10:]:
        print(f"  '{r['token']}': cos={r['cosine']:.4f} (solar_id={r['solar_id']}, glm_id={r['glm_id']})")

    # Statistical significance
    print("\n" + "=" * 70)
    print("STATISTICAL INTERPRETATION")
    print("=" * 70)

    mean_cos = np.mean(cosines)
    n = len(cosines)

    # Under null hypothesis (independent random embeddings), expected cosine ~ 0
    # Standard error ≈ 1/sqrt(hidden_size) ≈ 1/sqrt(4096) ≈ 0.0156
    se = 1.0 / np.sqrt(4096)
    z_score = mean_cos / se

    print(f"\n  Mean cosine: {mean_cos:.6f}")
    print(f"  Expected under null (random): ~0 ± {se:.4f}")
    print(f"  Z-score: {z_score:.2f}")

    if mean_cos > 0.5:
        print(f"\n  INTERPRETATION: VERY HIGH embedding similarity!")
        print(f"  This strongly suggests the embeddings share a common origin.")
    elif mean_cos > 0.1:
        print(f"\n  INTERPRETATION: Moderate embedding similarity.")
        print(f"  Some evidence of shared origin, but not conclusive alone.")
    else:
        print(f"\n  INTERPRETATION: Low embedding similarity.")
        print(f"  Embeddings appear to have diverged significantly.")

    # Visualization
    if HAS_MATPLOTLIB and cosines:
        print(f"\n[6] Creating visualization...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax = axes[0]
        ax.hist(cosines, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(cosines), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cosines):.3f}')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title(f'Embedding Cosine Similarity Distribution\n(n={len(cosines)} English tokens)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Scatter by token ID
        ax = axes[1]
        solar_ids = [r["solar_id"] for r in results if not np.isnan(r["cosine"])]
        ax.scatter(solar_ids, cosines, alpha=0.5, s=20, c='steelblue')
        ax.axhline(np.mean(cosines), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cosines):.3f}')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Solar Token ID')
        ax.set_ylabel('Cosine Similarity with GLM')
        ax.set_title('Embedding Similarity by Token ID')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "embedding_similarity.png"), dpi=150)
        plt.close()

        print(f"  Saved: {args.out_dir}/embedding_similarity.png")

    # Save results
    import csv
    csv_path = os.path.join(args.out_dir, "embedding_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["token", "solar_id", "glm_id", "same_id", "cosine"])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("EMBEDDING ANALYSIS CONCLUSION")
    print("=" * 70)

    if mean_cos > 0.8:
        verdict = "VERY STRONG evidence of embedding derivation"
    elif mean_cos > 0.5:
        verdict = "STRONG evidence of embedding derivation"
    elif mean_cos > 0.1:
        verdict = "MODERATE evidence - embeddings partially preserved"
    else:
        verdict = "WEAK evidence - embeddings significantly modified"

    print(f"""
  Mean embedding cosine for English tokens: {mean_cos:.4f}

  {verdict}

  Combined with LayerNorm analysis (cos=0.97), this provides
  {'additional confirmation' if mean_cos > 0.1 else 'contrasting evidence to'}
  that Solar-Open-100B derived from GLM-4.5-Air.
""")


if __name__ == "__main__":
    main()
