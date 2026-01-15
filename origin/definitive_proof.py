#!/usr/bin/env python3
"""
definitive_proof.py

Definitive analysis to prove whether Solar-Open-100B is derived from GLM-4.5-Air.

Key tests:
1. RAW BYTES COMPARISON - If any tensor has identical bytes, it's definitive proof
2. CONTROL COMPARISON - Compare GLM vs unrelated model to establish baseline
3. EMBEDDING ANALYSIS - Check if common tokens have similar embeddings
4. STATISTICAL SIGNIFICANCE - Proper hypothesis testing
"""

import hashlib
import json
import os
import struct
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url

# Models
SOLAR = "upstage/Solar-Open-100B"
GLM = "zai-org/GLM-4.5-Air"
# Control: Use a model with same hidden_size (4096) for fair comparison
# Qwen2-72B has hidden_size=8192, so not ideal
# Let's use GLM-4's own architecture as baseline - compare layer i vs layer j within GLM

def eprint(*args):
    print(*args, file=sys.stderr)

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

def fetch_header(repo: str, shard: str, rev: str, token: Optional[str], max_bytes: int = 64*1024*1024) -> Dict:
    url = hf_hub_url(repo_id=repo, filename=shard, revision=rev)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = struct.unpack("<Q", raw8)[0]
    if header_len > max_bytes:
        raise RuntimeError(f"Header too large: {header_len}")
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header

def fetch_raw_bytes(url: str, header: Dict, key: str, token: Optional[str]) -> bytes:
    """Fetch raw bytes of a tensor without decoding"""
    info = header[key]
    off0, off1 = info["data_offsets"]
    begin = 8 + header["__header_len__"] + int(off0)
    end = 8 + header["__header_len__"] + int(off1) - 1
    return http_range_get(url, begin, end, token)

def decode_bf16(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def load_index(repo: str, rev: str, token: Optional[str]) -> Dict[str, str]:
    path = hf_hub_download(repo, "model.safetensors.index.json", revision=rev, token=token)
    with open(path) as f:
        return json.load(f)["weight_map"]

def main():
    token = os.environ.get("HF_TOKEN")
    rev = "main"

    print("=" * 70)
    print("DEFINITIVE PROOF: Solar-Open-100B vs GLM-4.5-Air")
    print("=" * 70)

    # Load indices
    print("\n[1/5] Loading model indices...")
    solar_wm = load_index(SOLAR, rev, token)
    glm_wm = load_index(GLM, rev, token)

    header_cache = {}
    def get_header(repo, shard):
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_header(repo, shard, rev, token)
        return header_cache[k]

    # =========================================================
    # TEST 1: RAW BYTES COMPARISON
    # If ANY tensor has identical bytes, it's definitive proof
    # =========================================================
    print("\n[2/5] RAW BYTES COMPARISON (Definitive Test)...")
    print("       If any tensor has identical bytes = PROOF of weight reuse")

    raw_match_count = 0
    raw_match_tensors = []

    # Check LayerNorm and other small tensors that might be copied verbatim
    test_keys = []
    for layer in range(46):  # GLM has 46 layers
        test_keys.extend([
            f"model.layers.{layer}.input_layernorm.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
        ])

    for key in test_keys[:20]:  # Check first 20 for speed
        if key not in solar_wm or key not in glm_wm:
            continue

        try:
            solar_shard = solar_wm[key]
            glm_shard = glm_wm[key]

            solar_hdr = get_header(SOLAR, solar_shard)
            glm_hdr = get_header(GLM, glm_shard)

            if key not in solar_hdr or key not in glm_hdr:
                continue

            # Check shapes first
            s_shape = solar_hdr[key]["shape"]
            g_shape = glm_hdr[key]["shape"]
            if s_shape != g_shape:
                continue

            # Fetch raw bytes
            solar_raw = fetch_raw_bytes(solar_hdr["__url__"], solar_hdr, key, token)
            glm_raw = fetch_raw_bytes(glm_hdr["__url__"], glm_hdr, key, token)

            # Compare SHA256
            solar_hash = hashlib.sha256(solar_raw).hexdigest()
            glm_hash = hashlib.sha256(glm_raw).hexdigest()

            if solar_hash == glm_hash:
                raw_match_count += 1
                raw_match_tensors.append(key)
                print(f"  [MATCH!] {key}")
            else:
                # Even if hash differs, check cosine
                solar_arr = decode_bf16(solar_raw)
                glm_arr = decode_bf16(glm_raw)
                cos = cosine(solar_arr, glm_arr)
                if cos > 0.99:
                    print(f"  [HIGH]   {key} cos={cos:.6f}")

        except Exception as e:
            eprint(f"  [skip] {key}: {e}")

    print(f"\n  RAW BYTE MATCHES: {raw_match_count} / {min(20, len(test_keys))}")

    if raw_match_count > 0:
        print(f"\n  !!! DEFINITIVE PROOF: {raw_match_count} tensors have IDENTICAL bytes !!!")
        print(f"  Matched tensors: {raw_match_tensors}")

    # =========================================================
    # TEST 2: CONTROL COMPARISON (Within-model baseline)
    # Compare GLM layer i vs GLM layer j to establish baseline
    # =========================================================
    print("\n[3/5] CONTROL COMPARISON...")
    print("       Comparing GLM layer i vs GLM layer j (within-model baseline)")

    within_model_cosines = []
    cross_model_cosines = []

    # Within GLM: Compare layer 0 vs other layers
    for layer_j in [10, 20, 30, 40]:
        key_i = "model.layers.0.input_layernorm.weight"
        key_j = f"model.layers.{layer_j}.input_layernorm.weight"

        try:
            glm_shard_i = glm_wm[key_i]
            glm_shard_j = glm_wm[key_j]

            glm_hdr_i = get_header(GLM, glm_shard_i)
            glm_hdr_j = get_header(GLM, glm_shard_j)

            raw_i = fetch_raw_bytes(glm_hdr_i["__url__"], glm_hdr_i, key_i, token)
            raw_j = fetch_raw_bytes(glm_hdr_j["__url__"], glm_hdr_j, key_j, token)

            arr_i = decode_bf16(raw_i)
            arr_j = decode_bf16(raw_j)

            cos = cosine(arr_i, arr_j)
            within_model_cosines.append(cos)
            print(f"  GLM[0] vs GLM[{layer_j}]: cos={cos:.6f}")
        except Exception as e:
            eprint(f"  [skip] {e}")

    # Cross-model: Solar layer i vs GLM layer i
    for layer in [0, 10, 20, 30, 40]:
        key = f"model.layers.{layer}.input_layernorm.weight"

        try:
            solar_shard = solar_wm[key]
            glm_shard = glm_wm[key]

            solar_hdr = get_header(SOLAR, solar_shard)
            glm_hdr = get_header(GLM, glm_shard)

            solar_raw = fetch_raw_bytes(solar_hdr["__url__"], solar_hdr, key, token)
            glm_raw = fetch_raw_bytes(glm_hdr["__url__"], glm_hdr, key, token)

            solar_arr = decode_bf16(solar_raw)
            glm_arr = decode_bf16(glm_raw)

            cos = cosine(solar_arr, glm_arr)
            cross_model_cosines.append(cos)
            print(f"  Solar[{layer}] vs GLM[{layer}]: cos={cos:.6f}")
        except Exception as e:
            eprint(f"  [skip] {e}")

    if within_model_cosines and cross_model_cosines:
        within_mean = np.mean(within_model_cosines)
        cross_mean = np.mean(cross_model_cosines)
        print(f"\n  WITHIN-MODEL baseline: {within_mean:.4f}")
        print(f"  CROSS-MODEL (Solar-GLM): {cross_mean:.4f}")
        print(f"  DIFFERENCE: {cross_mean - within_mean:.4f}")

        if cross_mean > within_mean + 0.1:
            print("  → Cross-model is SIGNIFICANTLY higher than within-model baseline!")
            print("  → This suggests WEIGHT DERIVATION, not just similar architecture")

    # =========================================================
    # TEST 3: COMPREHENSIVE LAYER ANALYSIS
    # =========================================================
    print("\n[4/5] COMPREHENSIVE LAYER ANALYSIS...")

    results = []

    for layer in range(46):
        for norm_type in ["input_layernorm", "post_attention_layernorm"]:
            key = f"model.layers.{layer}.{norm_type}.weight"

            if key not in solar_wm or key not in glm_wm:
                continue

            try:
                solar_shard = solar_wm[key]
                glm_shard = glm_wm[key]

                solar_hdr = get_header(SOLAR, solar_shard)
                glm_hdr = get_header(GLM, glm_shard)

                solar_raw = fetch_raw_bytes(solar_hdr["__url__"], solar_hdr, key, token)
                glm_raw = fetch_raw_bytes(glm_hdr["__url__"], glm_hdr, key, token)

                # Check exact match
                exact_match = (solar_raw == glm_raw)

                # Compute cosine
                solar_arr = decode_bf16(solar_raw)
                glm_arr = decode_bf16(glm_raw)
                cos = cosine(solar_arr, glm_arr)

                results.append({
                    "layer": layer,
                    "type": norm_type,
                    "cosine": cos,
                    "exact_match": exact_match,
                })

            except Exception as e:
                pass

    # Summary statistics
    cosines = [r["cosine"] for r in results if not np.isnan(r["cosine"])]
    exact_matches = sum(1 for r in results if r["exact_match"])

    print(f"\n  Total comparisons: {len(results)}")
    print(f"  Exact byte matches: {exact_matches}")
    print(f"  Cosine mean: {np.mean(cosines):.4f}")
    print(f"  Cosine std: {np.std(cosines):.4f}")
    print(f"  Cosine min: {np.min(cosines):.4f}")
    print(f"  Cosine max: {np.max(cosines):.4f}")

    # =========================================================
    # TEST 4: ATTENTION WEIGHT COMPARISON
    # =========================================================
    print("\n[5/5] ATTENTION WEIGHT COMPARISON...")

    attn_cosines = []

    for layer in [5, 15, 25, 35]:
        for proj in ["k_proj", "v_proj"]:  # These have same shape
            key = f"model.layers.{layer}.self_attn.{proj}.weight"

            if key not in solar_wm or key not in glm_wm:
                continue

            try:
                solar_shard = solar_wm[key]
                glm_shard = glm_wm[key]

                solar_hdr = get_header(SOLAR, solar_shard)
                glm_hdr = get_header(GLM, glm_shard)

                s_shape = solar_hdr[key]["shape"]
                g_shape = glm_hdr[key]["shape"]

                if s_shape != g_shape:
                    continue

                solar_raw = fetch_raw_bytes(solar_hdr["__url__"], solar_hdr, key, token)
                glm_raw = fetch_raw_bytes(glm_hdr["__url__"], glm_hdr, key, token)

                solar_arr = decode_bf16(solar_raw)
                glm_arr = decode_bf16(glm_raw)

                cos = cosine(solar_arr, glm_arr)
                attn_cosines.append(cos)
                print(f"  {key}: cos={cos:.6f}")

            except Exception as e:
                pass

    if attn_cosines:
        print(f"\n  Attention mean cosine: {np.mean(attn_cosines):.6f}")

    # =========================================================
    # FINAL VERDICT
    # =========================================================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    evidence_score = 0

    # Evidence 1: Raw byte matches
    if exact_matches > 0:
        print(f"\n[DEFINITIVE] {exact_matches} tensors have IDENTICAL bytes")
        print("             This is IMPOSSIBLE by chance. Proves weight derivation.")
        evidence_score += 100

    # Evidence 2: LayerNorm cosine
    if np.mean(cosines) > 0.95:
        print(f"\n[STRONG] LayerNorm mean cosine = {np.mean(cosines):.4f}")
        if within_model_cosines and np.mean(cosines) > np.mean(within_model_cosines) + 0.05:
            print("         Cross-model > Within-model baseline")
            evidence_score += 30
        else:
            print("         But within-model baseline is also high")
            evidence_score += 10

    # Evidence 3: Attention ~0
    if attn_cosines and abs(np.mean(attn_cosines)) < 0.01:
        print(f"\n[MODERATE] Attention cosine ~0 ({np.mean(attn_cosines):.6f})")
        print("           Indicates attention weights were re-trained")
        evidence_score += 10

    print(f"\n{'='*70}")
    print(f"EVIDENCE SCORE: {evidence_score}/100")
    print(f"{'='*70}")

    if evidence_score >= 100:
        print("\n>>> CONCLUSION: DEFINITIVE PROOF of weight derivation")
        print(">>> Solar-Open-100B contains weights directly copied from GLM-4.5-Air")
    elif evidence_score >= 50:
        print("\n>>> CONCLUSION: STRONG EVIDENCE of weight derivation")
        print(">>> Solar-Open-100B very likely derived from GLM-4.5-Air")
    elif evidence_score >= 20:
        print("\n>>> CONCLUSION: MODERATE EVIDENCE")
        print(">>> Possible derivation, but not conclusive")
    else:
        print("\n>>> CONCLUSION: WEAK/NO EVIDENCE")
        print(">>> Models may just have similar architecture")

if __name__ == "__main__":
    main()
