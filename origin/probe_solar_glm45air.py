#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probe whether Solar-Open-100B looks like "GLM-4.5-Air with num_nextn_predict_layers=0"
and whether weights show suspicious similarity.

What it does:
  A) Download & diff config.json (architecture-level claims)
  B) Download sharded safetensors index, scan key patterns (nextn/MTP hints)
  C) Read safetensors headers via HTTP Range (no full download)
  D) Compare:
     - shapes of critical tensors (k_proj/v_proj/q_proj/o_proj, router, norms)
     - value similarity for shape-compatible tensors using sampled windows (Range)
     - optional "q_proj truncation" test if head count differs (small == prefix rows of big)

Requirements:
  pip install requests numpy

Optional:
  export HF_TOKEN=...   # if gated (often not needed)

Usage:
  python probe_solar_glm45air.py \
    --solar upstage/Solar-Open-100B \
    --glm   zai-org/GLM-4.5-Air \
    --layers 0-45 \
    --windows 3 \
    --chunk_elems 262144 \
    --try_truncation \
    --out report.csv
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests


DTYPE_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

FLOAT_DTYPES = {"BF16", "F16", "F32"}


def hf_url(repo_id: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _headers(token: Optional[str], extra: Optional[dict] = None) -> dict:
    h = {"Accept-Encoding": "identity"}  # 중요: Range 오프셋 안정성을 위해 gzip 피함
    if token:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update(extra)
    return h


def http_get(url: str, token: Optional[str]) -> Optional[bytes]:
    r = requests.get(url, headers=_headers(token), allow_redirects=True, timeout=120)
    if r.status_code == 200:
        return r.content
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
    """
    Range is inclusive: bytes=start-end
    """
    r = requests.get(
        url,
        headers=_headers(token, {"Range": f"bytes={start}-{end}"}),
        allow_redirects=True,
        timeout=120,
    )
    if r.status_code == 206:
        return r.content
    # 일부 서버/리다이렉트 조합에서 200으로 오는데 길이가 딱 맞는 경우만 허용
    expected = end - start + 1
    if r.status_code == 200 and len(r.content) == expected:
        return r.content
    raise RuntimeError(f"Range GET failed: HTTP {r.status_code} len={len(r.content)} expected={expected}")


@dataclass
class TensorInfo:
    dtype: str
    shape: Tuple[int, ...]
    data_offsets: Tuple[int, int]  # [begin, end) relative to data section


@dataclass
class SafetensorsHeader:
    header_len: int
    base_data_offset: int  # 8 + header_len
    tensors: Dict[str, TensorInfo]


class HeaderCache:
    def __init__(self):
        self._cache: Dict[str, SafetensorsHeader] = {}

    def get(self, url: str) -> Optional[SafetensorsHeader]:
        return self._cache.get(url)

    def set(self, url: str, hdr: SafetensorsHeader) -> None:
        self._cache[url] = hdr


def parse_safetensors_header(url: str, token: Optional[str], cache: HeaderCache) -> SafetensorsHeader:
    cached = cache.get(url)
    if cached:
        return cached

    b0 = http_range_get(url, 0, 7, token)
    header_len = struct.unpack("<Q", b0)[0]
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header_str = hb.decode("utf-8").strip()
    header_json = json.loads(header_str)

    tensors: Dict[str, TensorInfo] = {}
    for k, v in header_json.items():
        if k == "__metadata__":
            continue
        tensors[k] = TensorInfo(
            dtype=v["dtype"],
            shape=tuple(int(x) for x in v["shape"]),
            data_offsets=(int(v["data_offsets"][0]), int(v["data_offsets"][1])),
        )

    hdr = SafetensorsHeader(
        header_len=int(header_len),
        base_data_offset=8 + int(header_len),
        tensors=tensors,
    )
    cache.set(url, hdr)
    return hdr


def prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for x in shape:
        p *= int(x)
    return int(p)


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


def stable_u64(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    nx = np.linalg.norm(x64)
    ny = np.linalg.norm(y64)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x64, y64) / (nx * ny))


def rel_rmse(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    mse = np.mean((x64 - y64) ** 2)
    denom = np.sqrt(np.mean(x64 ** 2)) + 1e-12
    return float(np.sqrt(mse) / denom)


class RemoteRepo:
    def __init__(self, repo_id: str, revision: str, token: Optional[str]):
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.header_cache = HeaderCache()

        self.config = self._load_config()
        self.weight_map, self.index_file = self._load_weight_map()

    def _load_config(self) -> dict:
        url = hf_url(self.repo_id, self.revision, "config.json")
        cfg = http_get_json(url, self.token)
        return cfg or {}

    def _load_weight_map(self) -> Tuple[Dict[str, str], str]:
        candidates = [
            "model.safetensors.index.json",
            "pytorch_model.safetensors.index.json",
        ]
        for name in candidates:
            url = hf_url(self.repo_id, self.revision, name)
            js = http_get_json(url, self.token)
            if js and isinstance(js, dict) and "weight_map" in js:
                wm = js["weight_map"]
                if isinstance(wm, dict):
                    return wm, name

        # unsharded fallback (rare for huge models). Try model.safetensors header scan
        for single in ["model.safetensors", "pytorch_model.safetensors"]:
            url = hf_url(self.repo_id, self.revision, single)
            try:
                hdr = parse_safetensors_header(url, self.token, self.header_cache)
                wm = {k: single for k in hdr.tensors.keys()}
                return wm, single
            except Exception:
                continue

        raise RuntimeError(f"Could not find safetensors index in {self.repo_id}")

    def file_url(self, filename: str) -> str:
        return hf_url(self.repo_id, self.revision, filename)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def tensor_info(self, key: str) -> Tuple[str, Tuple[int, ...], str, SafetensorsHeader, TensorInfo]:
        """
        Returns: (shard_filename, shape, dtype, shard_header, tensor_info)
        """
        shard = self.weight_map[key]
        url = self.file_url(shard)
        hdr = parse_safetensors_header(url, self.token, self.header_cache)
        if key not in hdr.tensors:
            raise KeyError(f"{key} not found inside shard header: {shard}")
        info = hdr.tensors[key]
        return shard, info.shape, info.dtype, hdr, info

    def sample_windows(
        self,
        key: str,
        chunk_elems: int,
        windows: int,
        seed: int,
        # Optional: override viewing region inside the tensor (for truncation test)
        view_numel: Optional[int] = None,
        view_data_begin_rel: Optional[int] = None,
    ) -> Tuple[str, np.ndarray, int]:
        """
        Returns: (raw_bytes_sha256, values_f32_concat, total_elems_sampled)

        Sampling uses contiguous windows to keep Range requests small.
        """
        shard, shape, dtype, hdr, info = self.tensor_info(key)
        if dtype not in FLOAT_DTYPES:
            raise ValueError(f"dtype {dtype} not supported for numeric sampling")

        elem_size = DTYPE_SIZES[dtype]
        full_numel = prod(shape)

        # tensor data region begin relative to data section:
        data_begin_rel = info.data_offsets[0]
        data_end_rel = info.data_offsets[1]
        tensor_bytes = data_end_rel - data_begin_rel
        if tensor_bytes != full_numel * elem_size:
            # Usually should match
            pass

        # Apply optional view (e.g., big tensor prefix rows)
        if view_numel is None:
            view_numel = full_numel
        if view_data_begin_rel is None:
            view_data_begin_rel = data_begin_rel

        # windows * chunk_elems might exceed view_numel; clamp
        if view_numel <= 0:
            raise ValueError("view_numel must be positive")

        raw_hasher = hashlib.sha256()
        vals = []
        total = 0

        for w in range(max(1, windows)):
            if view_numel <= chunk_elems:
                start_elem = 0
                n = view_numel
            else:
                # deterministic per key/window
                s = stable_u64(f"{key}|{seed}|{w}")
                start_elem = int(s % (view_numel - chunk_elems))
                n = chunk_elems

            abs_start = hdr.base_data_offset + view_data_begin_rel + start_elem * elem_size
            abs_end_incl = abs_start + n * elem_size - 1

            shard_url = self.file_url(shard)
            buf = http_range_get(shard_url, abs_start, abs_end_incl, self.token)

            raw_hasher.update(buf)
            v = bytes_to_f32(buf, dtype)
            vals.append(v)
            total += len(v)

        raw_hash = raw_hasher.hexdigest()
        out = np.concatenate(vals, axis=0) if len(vals) > 1 else vals[0]
        return raw_hash, out, total


def parse_layers_arg(s: str) -> List[int]:
    """
    Accepts formats:
      "0-45"
      "0,1,2,10,20"
      "0-10,20,30-35"
    """
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    out = sorted(set(out))
    return out


def keyscan_summary(weight_map: Dict[str, str]) -> dict:
    pat_layer = re.compile(r"^model\.layers\.(\d+)\.")
    layers = []
    for k in weight_map.keys():
        m = pat_layer.match(k)
        if m:
            layers.append(int(m.group(1)))
    layers = sorted(layers)
    max_layer = max(layers) if layers else None

    # nextn/mtp keyword scan
    kws = ["nextn", "mtp", "predict", "gmask", "sop"]
    kw_hits = {kw: 0 for kw in kws}
    for k in weight_map.keys():
        lk = k.lower()
        for kw in kws:
            if kw in lk:
                kw_hits[kw] += 1

    # count by suspicious layer indices
    suspicious = {}
    for idx in [46, 92]:
        pref = f"model.layers.{idx}."
        suspicious[idx] = sum(1 for k in weight_map.keys() if k.startswith(pref))

    return {
        "num_keys": len(weight_map),
        "layer_min": min(layers) if layers else None,
        "layer_max": max_layer,
        "num_layer_indices": len(set(layers)) if layers else 0,
        "suspicious_layer_key_counts": suspicious,
        "keyword_hits": kw_hits,
    }


def print_config_diff(a: RemoteRepo, b: RemoteRepo) -> None:
    # Fields most relevant to "GLM4.5-Air with nextn=0" claim
    fields = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "rope_theta",
        "max_position_embeddings",
        "vocab_size",
        "num_nextn_predict_layers",
    ]
    print("\n[CONFIG DIFF] (selected fields)\n")
    for f in fields:
        va = a.config.get(f, None)
        vb = b.config.get(f, None)
        mark = "==" if va == vb else "!="
        print(f"{f:28s} Solar={va!r:>12}  GLM={vb!r:>12}   {mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solar", default="upstage/Solar-Open-100B")
    ap.add_argument("--glm", default="zai-org/GLM-4.5-Air")
    ap.add_argument("--rev_solar", default="main")
    ap.add_argument("--rev_glm", default="main")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (optional)")
    ap.add_argument("--layers", default="0-45", help='e.g. "0-45" or "0,1,2,10,20"')
    ap.add_argument("--windows", type=int, default=3, help="number of windows to sample per tensor")
    ap.add_argument("--chunk_elems", type=int, default=262144, help="elements per window (contiguous)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--try_truncation", action="store_true", help="Try q_proj prefix-row truncation test if shapes mismatch")
    ap.add_argument("--out", default="report.csv")
    args = ap.parse_args()

    layers = parse_layers_arg(args.layers)

    solar = RemoteRepo(args.solar, args.rev_solar, args.token)
    glm = RemoteRepo(args.glm, args.rev_glm, args.token)

    print_config_diff(solar, glm)

    # Index scans
    s_sum = keyscan_summary(solar.weight_map)
    g_sum = keyscan_summary(glm.weight_map)
    print("\n[INDEX SUMMARY]\n")
    print(f"Solar index file: {solar.index_file} | keys={s_sum['num_keys']} | layer_idx_count={s_sum['num_layer_indices']} | layer_max={s_sum['layer_max']}")
    print(f"GLM   index file: {glm.index_file}   | keys={g_sum['num_keys']} | layer_idx_count={g_sum['num_layer_indices']} | layer_max={g_sum['layer_max']}")
    print(f"Solar suspicious layer key counts: {s_sum['suspicious_layer_key_counts']}")
    print(f"GLM   suspicious layer key counts: {g_sum['suspicious_layer_key_counts']}")
    print(f"Solar keyword hits: {s_sum['keyword_hits']}")
    print(f"GLM   keyword hits: {g_sum['keyword_hits']}")

    # Target keys to test.
    # NOTE: k_proj/v_proj + router + norms are most robust against q-head count differences.
    key_templates = [
        "model.layers.{L}.self_attn.k_proj.weight",
        "model.layers.{L}.self_attn.v_proj.weight",
        "model.layers.{L}.self_attn.q_proj.weight",
        "model.layers.{L}.self_attn.o_proj.weight",
        "model.layers.{L}.input_layernorm.weight",
        "model.layers.{L}.post_attention_layernorm.weight",
        "model.layers.{L}.mlp.gate.weight",
        "model.layers.{L}.mlp.gate.e_score_correction_bias",
    ]

    rows = []

    print("\n[SHAPE + VALUE SAMPLING]\n")
    for L in layers:
        for tmpl in key_templates:
            key = tmpl.format(L=L)

            if not (solar.has(key) and glm.has(key)):
                rows.append({
                    "layer": L,
                    "tensor": key,
                    "status": "missing_in_one_repo",
                    "solar_dtype": "",
                    "glm_dtype": "",
                    "solar_shape": "",
                    "glm_shape": "",
                    "raw_hash_equal": "",
                    "cosine": "",
                    "rel_rmse": "",
                    "z_cos": "",
                    "note": "",
                })
                continue

            try:
                _, s_shape, s_dtype, _, _ = solar.tensor_info(key)
                _, g_shape, g_dtype, _, _ = glm.tensor_info(key)

                if s_shape != g_shape or s_dtype != g_dtype:
                    # Try optional truncation test for q_proj
                    note = ""
                    if args.try_truncation and key.endswith("self_attn.q_proj.weight"):
                        # If in_features match and GLM has more out_features, test if Solar equals prefix rows of GLM
                        # weight shape is (out_features, in_features)
                        if len(s_shape) == 2 and len(g_shape) == 2 and s_shape[1] == g_shape[1] and s_shape[0] < g_shape[0]:
                            # view_numel = s_out * in
                            view_numel = s_shape[0] * s_shape[1]
                            # big tensor prefix rows start at its data_begin_rel (no extra offset)
                            # so view_data_begin_rel is just original data_begin_rel
                            # We'll sample big tensor in that view range and compare with Solar full tensor.
                            # We need big tensor's data_begin_rel:
                            _, _, _, g_hdr, g_info = glm.tensor_info(key)
                            g_view_begin_rel = g_info.data_offsets[0]
                            s_raw, s_vals, n1 = solar.sample_windows(key, args.chunk_elems, args.windows, args.seed)
                            g_raw, g_vals, n2 = glm.sample_windows(
                                key,
                                args.chunk_elems,
                                args.windows,
                                args.seed,
                                view_numel=view_numel,
                                view_data_begin_rel=g_view_begin_rel,
                            )
                            n = min(n1, n2)
                            s_vals = s_vals[:n]
                            g_vals = g_vals[:n]
                            cos = cosine(s_vals, g_vals)
                            rr = rel_rmse(s_vals, g_vals)
                            z = cos * math.sqrt(n) if not math.isnan(cos) else float("nan")
                            rows.append({
                                "layer": L,
                                "tensor": key,
                                "status": "shape_mismatch_but_trunc_test_ok",
                                "solar_dtype": s_dtype,
                                "glm_dtype": g_dtype,
                                "solar_shape": "x".join(map(str, s_shape)),
                                "glm_shape": "x".join(map(str, g_shape)),
                                "raw_hash_equal": str(s_raw == g_raw),
                                "cosine": cos,
                                "rel_rmse": rr,
                                "z_cos": z,
                                "note": "Compared Solar vs GLM prefix-rows slice",
                            })
                            print(f"[TRUNC_OK] L{L:02d} cos={cos:+.6f} z~{z:+.2f}  {key}")
                            continue
                        else:
                            note = "q_proj truncation not applicable"

                    rows.append({
                        "layer": L,
                        "tensor": key,
                        "status": "shape_or_dtype_mismatch",
                        "solar_dtype": s_dtype,
                        "glm_dtype": g_dtype,
                        "solar_shape": "x".join(map(str, s_shape)),
                        "glm_shape": "x".join(map(str, g_shape)),
                        "raw_hash_equal": "",
                        "cosine": "",
                        "rel_rmse": "",
                        "z_cos": "",
                        "note": note,
                    })
                    print(f"[MISMATCH] L{L:02d} {key} | Solar {s_shape}/{s_dtype} vs GLM {g_shape}/{g_dtype}")
                    continue

                # shape match => sample values
                s_raw, s_vals, n1 = solar.sample_windows(key, args.chunk_elems, args.windows, args.seed)
                g_raw, g_vals, n2 = glm.sample_windows(key, args.chunk_elems, args.windows, args.seed)
                n = min(n1, n2)
                s_vals = s_vals[:n]
                g_vals = g_vals[:n]

                cos = cosine(s_vals, g_vals)
                rr = rel_rmse(s_vals, g_vals)
                z = cos * math.sqrt(n) if not math.isnan(cos) else float("nan")

                rows.append({
                    "layer": L,
                    "tensor": key,
                    "status": "ok",
                    "solar_dtype": s_dtype,
                    "glm_dtype": g_dtype,
                    "solar_shape": "x".join(map(str, s_shape)),
                    "glm_shape": "x".join(map(str, g_shape)),
                    "raw_hash_equal": str(s_raw == g_raw),
                    "cosine": cos,
                    "rel_rmse": rr,
                    "z_cos": z,
                    "note": "",
                })

                print(f"[OK] L{L:02d} cos={cos:+.6f} rr={rr:.6f} z~{z:+.2f} raw_equal={s_raw==g_raw}  {key}")

            except Exception as e:
                rows.append({
                    "layer": L,
                    "tensor": key,
                    "status": f"error:{type(e).__name__}",
                    "solar_dtype": "",
                    "glm_dtype": "",
                    "solar_shape": "",
                    "glm_shape": "",
                    "raw_hash_equal": "",
                    "cosine": "",
                    "rel_rmse": "",
                    "z_cos": "",
                    "note": str(e),
                })
                print(f"[ERROR] L{L:02d} {key}: {e}")

    # Save CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {args.out}")
    print("\nHow to interpret:")
    print("- If many large tensors show raw_hash_equal=True or cosine≈1 across multiple layers => extremely strong evidence of reuse/identity.")
    print("- If cosines stay ~0 with small magnitude (e.g. |cos| < 0.003 for n≈262k) across many tensors/layers => no evidence of weight reuse.")
    print("- k_proj/v_proj/router/norm are the most comparable even if q-head count differs.")
    print("- If q_proj truncation test (shape_mismatch_but_trunc_test_ok) yields very high cosine repeatedly => strong evidence of truncation-based derivation.")


if __name__ == "__main__":
    main()
