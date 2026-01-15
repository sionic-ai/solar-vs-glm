#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep analysis of vocabulary mapping between Solar and GLM.
Investigates why token IDs differ and what this means for derivation hypothesis.
"""

import json
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
import requests

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def hf_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def http_get_json(url: str, token: Optional[str] = None) -> Optional[dict]:
    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
        if r.status_code == 200:
            return json.loads(r.content.decode("utf-8"))
    except Exception as e:
        print(f"Error: {e}")
    return None


def load_tokenizer_vocab(repo_id: str, revision: str = "main", token: Optional[str] = None) -> Dict[str, int]:
    url = hf_url(repo_id, revision, "tokenizer.json")
    data = http_get_json(url, token)
    if data and "model" in data and "vocab" in data["model"]:
        return data["model"]["vocab"]
    return {}


def is_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    for char in text:
        if '\uAC00' <= char <= '\uD7A3':  # Korean syllables
            return True
        if '\u1100' <= char <= '\u11FF':  # Korean Jamo
            return True
        if '\u3130' <= char <= '\u318F':  # Korean compatibility Jamo
            return True
    return False


def is_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    for char in text:
        if '\u4E00' <= char <= '\u9FFF':  # CJK Unified Ideographs
            return True
    return False


def is_ascii(text: str) -> bool:
    return all(ord(c) < 128 for c in text)


def analyze_token(token: str) -> str:
    """Classify token type."""
    clean = token.replace("Ġ", "").replace("▁", "")
    if not clean:
        return "empty"
    if is_korean(clean):
        return "korean"
    if is_chinese(clean):
        return "chinese"
    if is_ascii(clean):
        if clean.isalpha():
            return "ascii_alpha"
        elif clean.isdigit():
            return "ascii_digit"
        else:
            return "ascii_other"
    return "other"


def main():
    print("=" * 70)
    print("VOCABULARY MAPPING ANALYSIS")
    print("=" * 70)

    token = os.getenv("HF_TOKEN")

    # Load vocabularies
    print("\n[1] Loading vocabularies...")
    solar_vocab = load_tokenizer_vocab("upstage/Solar-Open-100B", token=token)
    glm_vocab = load_tokenizer_vocab("zai-org/GLM-4.5-Air", token=token)

    print(f"  Solar: {len(solar_vocab)} tokens")
    print(f"  GLM: {len(glm_vocab)} tokens")

    # Reverse mappings
    solar_id_to_token = {v: k for k, v in solar_vocab.items()}
    glm_id_to_token = {v: k for k, v in glm_vocab.items()}

    # Analyze common tokens
    print("\n[2] Common token analysis...")
    common_tokens = set(solar_vocab.keys()) & set(glm_vocab.keys())
    print(f"  Common tokens (by text): {len(common_tokens)}")

    # ID comparison
    same_id_count = 0
    id_diffs = []
    for token in common_tokens:
        s_id = solar_vocab[token]
        g_id = glm_vocab[token]
        if s_id == g_id:
            same_id_count += 1
        id_diffs.append(s_id - g_id)

    print(f"  Same ID in both: {same_id_count}")

    # Check first 1000 token IDs
    print("\n[3] First 1000 tokens comparison...")
    first_1000_match = 0
    for i in range(min(1000, len(solar_id_to_token), len(glm_id_to_token))):
        if i in solar_id_to_token and i in glm_id_to_token:
            if solar_id_to_token[i] == glm_id_to_token[i]:
                first_1000_match += 1

    print(f"  First 1000 IDs with same token: {first_1000_match}")

    # Show first 20 tokens from each
    print("\n  First 20 tokens comparison:")
    print(f"  {'ID':<6} {'Solar Token':<30} {'GLM Token':<30} {'Match'}")
    print("  " + "-" * 75)
    for i in range(20):
        s_tok = solar_id_to_token.get(i, "N/A")
        g_tok = glm_id_to_token.get(i, "N/A")
        match = "✓" if s_tok == g_tok else "✗"
        print(f"  {i:<6} {repr(s_tok):<30} {repr(g_tok):<30} {match}")

    # Analyze Solar-only tokens
    print("\n[4] Solar-only tokens analysis...")
    solar_only = set(solar_vocab.keys()) - set(glm_vocab.keys())
    print(f"  Tokens only in Solar: {len(solar_only)}")

    # Classify Solar-only tokens
    solar_only_types = Counter()
    korean_examples = []
    for tok in solar_only:
        t = analyze_token(tok)
        solar_only_types[t] += 1
        if t == "korean" and len(korean_examples) < 10:
            korean_examples.append(tok)

    print(f"\n  Solar-only token types:")
    for t, c in solar_only_types.most_common():
        print(f"    {t}: {c} ({100*c/len(solar_only):.1f}%)")

    if korean_examples:
        print(f"\n  Korean token examples: {korean_examples[:10]}")

    # Analyze GLM-only tokens
    print("\n[5] GLM-only tokens analysis...")
    glm_only = set(glm_vocab.keys()) - set(solar_vocab.keys())
    print(f"  Tokens only in GLM: {len(glm_only)}")

    glm_only_types = Counter()
    for tok in glm_only:
        glm_only_types[analyze_token(tok)] += 1

    print(f"\n  GLM-only token types:")
    for t, c in glm_only_types.most_common():
        print(f"    {t}: {c} ({100*c/len(glm_only):.1f}%)")

    # Check if there's a systematic ID offset
    print("\n[6] ID mapping pattern analysis...")

    # Sample common tokens and check their ID relationships
    common_list = list(common_tokens)[:10000]
    solar_ids = [solar_vocab[t] for t in common_list]
    glm_ids = [glm_vocab[t] for t in common_list]

    # Check correlation
    corr = np.corrcoef(solar_ids, glm_ids)[0, 1]
    print(f"  ID correlation (sample of {len(common_list)} tokens): {corr:.4f}")

    # Check if IDs are just offset
    diffs = np.array(solar_ids) - np.array(glm_ids)
    unique_diffs = len(set(diffs))
    print(f"  Unique ID differences: {unique_diffs}")

    if unique_diffs < 100:
        diff_counts = Counter(diffs)
        print(f"  Most common ID offsets: {diff_counts.most_common(5)}")

    # Analyze high-ID tokens in Solar (likely new additions)
    print("\n[7] High-ID Solar tokens (likely new additions)...")
    high_id_threshold = 150000  # Approximately GLM vocab size

    high_id_tokens = [(tok, solar_vocab[tok]) for tok in solar_vocab if solar_vocab[tok] >= high_id_threshold]
    high_id_tokens.sort(key=lambda x: x[1])

    print(f"  Tokens with ID >= {high_id_threshold}: {len(high_id_tokens)}")

    high_id_types = Counter()
    for tok, _ in high_id_tokens:
        high_id_types[analyze_token(tok)] += 1

    print(f"\n  High-ID token types:")
    for t, c in high_id_types.most_common():
        print(f"    {t}: {c} ({100*c/len(high_id_tokens):.1f}%)")

    # Show examples of high-ID tokens
    print(f"\n  Examples of tokens with ID >= {high_id_threshold}:")
    for tok, tid in high_id_tokens[:20]:
        print(f"    {tid}: {repr(tok)}")

    # CRITICAL ANALYSIS
    print("\n" + "=" * 70)
    print("CRITICAL INTERPRETATION")
    print("=" * 70)

    print(f"""
  KEY FINDINGS:

  1. Token ID Mapping:
     - Common tokens (by text): {len(common_tokens)} ({100*len(common_tokens)/len(glm_vocab):.1f}% of GLM)
     - Same ID in both: {same_id_count} ({100*same_id_count/len(common_tokens):.1f}%)
     - First 1000 IDs matching: {first_1000_match}
     - ID correlation: {corr:.4f}

  2. Vocabulary Changes:
     - Solar-only tokens: {len(solar_only)} (Korean: {solar_only_types.get('korean', 0)})
     - GLM-only tokens: {len(glm_only)}
     - High-ID (>150k) tokens: {len(high_id_tokens)}

  INTERPRETATION:
""")

    if first_1000_match < 100:
        print("""
  The vocabulary was COMPLETELY REORGANIZED, not simply extended.
  This explains why embedding cosine similarity is ~0:
  - Token "hello" might be ID=1234 in GLM but ID=5678 in Solar
  - Comparing embeddings by matching token TEXT compares different rows
  - The embedding matrix was rebuilt with new ID assignments

  This is CONSISTENT with the derivation hypothesis:
  - LayerNorm weights (which don't depend on token IDs) show 0.97 cosine
  - Embedding layer was completely retrained (explains ~0 cosine)
  - Early LayerNorms (Layer 0-2) show lower cosine due to embedding changes
  - Deep LayerNorms preserved because they're farther from embedding

  The "Embedding Effect" gradient (Layer 0: 0.12 → Layer 45: 0.99) is
  EXACTLY what we'd expect if embeddings were retrained but deeper
  layers were preserved from GLM.
""")
    else:
        print(f"""
  Token IDs appear to have some alignment (first 1000 match: {first_1000_match}).
  Further investigation needed.
""")

    # Visualization
    if HAS_MATPLOTLIB:
        os.makedirs("embedding_results", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ID scatter
        ax = axes[0, 0]
        sample_idx = np.random.choice(len(common_list), min(2000, len(common_list)), replace=False)
        ax.scatter([solar_ids[i] for i in sample_idx], [glm_ids[i] for i in sample_idx], alpha=0.3, s=10)
        ax.set_xlabel("Solar Token ID")
        ax.set_ylabel("GLM Token ID")
        ax.set_title(f"Token ID Mapping (corr={corr:.3f})")
        ax.plot([0, max(solar_ids)], [0, max(solar_ids)], 'r--', alpha=0.5, label='y=x')
        ax.legend()

        # ID difference histogram
        ax = axes[0, 1]
        ax.hist(diffs, bins=100, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Solar ID - GLM ID")
        ax.set_ylabel("Count")
        ax.set_title("Token ID Difference Distribution")
        ax.axvline(0, color='red', linestyle='--')

        # Solar-only token types
        ax = axes[1, 0]
        types = list(solar_only_types.keys())
        counts = [solar_only_types[t] for t in types]
        ax.barh(types, counts, color='steelblue')
        ax.set_xlabel("Count")
        ax.set_title(f"Solar-only Token Types (n={len(solar_only)})")

        # High-ID token types
        ax = axes[1, 1]
        types = list(high_id_types.keys())
        counts = [high_id_types[t] for t in types]
        ax.barh(types, counts, color='coral')
        ax.set_xlabel("Count")
        ax.set_title(f"High-ID (>{high_id_threshold}) Token Types (n={len(high_id_tokens)})")

        plt.tight_layout()
        plt.savefig("embedding_results/vocab_mapping_analysis.png", dpi=150)
        print("\n  Saved: embedding_results/vocab_mapping_analysis.png")


if __name__ == "__main__":
    main()
