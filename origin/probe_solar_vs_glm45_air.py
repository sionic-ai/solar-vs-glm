#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_solar_vs_glm45_air.py

End-to-end provenance-ish probe between:
  A: upstage/Solar-Open-100B
  B: zai-org/GLM-4.5-Air

What it does:
  (1) Compare config.json (key fields)
  (2) Compare tokenizer.json fingerprints:
      - vocab size, merges length
      - merges sha256 (first 1000 + full)
      - token set overlap (Jaccard)
      - ASCII token offset histogram & longest contiguous match run
  (3) Layerwise weight similarity visualization for small/diagnostic tensors:
      - per-layer norms (input/pre + post)
      - per-layer router/gate weights (if found)
      - cosine + pearson
      - optional: expert-row alignment (Hungarian, requires scipy)
      - optional: layer alignment (Hungarian, requires scipy)

Implementation detail:
  - Uses model.safetensors.index.json to locate which shard contains a tensor key.
  - Uses HTTP Range requests to read safetensors header and only the selected tensor bytes.
  - Avoids downloading whole shards.

Requirements:
  pip install -U huggingface_hub requests numpy matplotlib
  pip install -U scipy   (optional, for optimal assignment)

Env:
  export HF_TOKEN=hf_xxx  # if needed
"""

import argparse
import dataclasses
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


# ----------------------------
# Optional SciPy (assignment)
# ----------------------------
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------------
# Utilities
# ----------------------------

ASCII_RE = re.compile(r"^[\x00-\x7F]+$")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def sha256_lines(lines: List[str], limit: Optional[int] = None) -> str:
    h = hashlib.sha256()
    n = len(lines) if limit is None else min(len(lines), limit)
    for i in range(n):
        h.update((lines[i] + "\n").encode("utf-8"))
    return h.hexdigest()

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(path: str, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            parts = []
            for c in cols:
                v = r.get(c, "")
                s = str(v)
                # keep CSV simple
                s = s.replace("\n", " ").replace("\r", " ").replace(",", " ")
                parts.append(s)
            f.write(",".join(parts) + "\n")


# ----------------------------
# Download HF files
# ----------------------------

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


# ----------------------------
# Tokenizer analysis
# ----------------------------

def inv_vocab(tok_json: Dict[str, Any]) -> List[str]:
    vocab = tok_json["model"]["vocab"]  # token -> id
    inv = [None] * len(vocab)
    for t, i in vocab.items():
        inv[i] = t
    return inv  # type: ignore

def merges_list(tok_json: Dict[str, Any]) -> List[str]:
    merges = tok_json["model"].get("merges", []) or []
    out = []
    for m in merges:
        if isinstance(m, str):
            out.append(m)
        elif isinstance(m, (list, tuple)):
            out.append(" ".join(map(str, m)))
        else:
            out.append(str(m))
    return out

def tokenizer_fingerprint(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    tj = load_tokenizer_json(repo_id, revision, token)
    inv = inv_vocab(tj)
    merges = merges_list(tj)
    return {
        "model_id": repo_id,
        "tokenizer_model_type": tj.get("model", {}).get("type", None),
        "vocab_size": len(inv),
        "merges_len": len(merges),
        "merges_sha256_first1000": sha256_lines(merges, limit=1000),
        "merges_sha256_all": sha256_lines(merges, limit=None),
    }

def offset_histogram_ascii(invA: List[str], invB: List[str], max_tokens: int = 200_000) -> Counter:
    mapB = {t: i for i, t in enumerate(invB) if t is not None}
    c = Counter()
    n = min(len(invA), max_tokens)
    for i in range(n):
        t = invA[i]
        if t is None:
            continue
        if not ASCII_RE.match(t):
            continue
        j = mapB.get(t)
        if j is None:
            continue
        c[j - i] += 1
    return c

def longest_contiguous_match(invA: List[str], invB: List[str], offset: int) -> Tuple[int, Optional[int]]:
    """
    Match when invA[i] == invB[i + offset]
    Returns (best_run_len, start_index_in_A)
    """
    startA = max(0, -offset)
    endA = min(len(invA), len(invB) - offset)
    if endA <= startA:
        return 0, None

    best = 0
    best_start = None
    run = 0
    for i in range(startA, endA):
        if invA[i] == invB[i + offset]:
            run += 1
            if run > best:
                best = run
                best_start = i - run + 1
        else:
            run = 0
    return best, best_start

def tokenizer_compare(repoA: str, repoB: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    tjA = load_tokenizer_json(repoA, revision, token)
    tjB = load_tokenizer_json(repoB, revision, token)
    invA = inv_vocab(tjA)
    invB = inv_vocab(tjB)

    setA = set(tjA["model"]["vocab"].keys())
    setB = set(tjB["model"]["vocab"].keys())
    inter = len(setA & setB)
    union = len(setA | setB)

    same_id = sum(1 for i in range(min(len(invA), len(invB))) if invA[i] == invB[i])

    hist = offset_histogram_ascii(invA, invB)
    top_offsets = hist.most_common(10)
    best_offset = top_offsets[0][0] if top_offsets else 0
    run_len, run_start = longest_contiguous_match(invA, invB, best_offset)

    sampleA = invA[run_start:run_start+8] if (run_start is not None and run_len > 0) else []
    sampleB = invB[run_start+best_offset:run_start+best_offset+8] if (run_start is not None and run_len > 0) else []

    return {
        "A_fingerprint": tokenizer_fingerprint(repoA, revision, token),
        "B_fingerprint": tokenizer_fingerprint(repoB, revision, token),
        "token_set_overlap": {
            "intersection": inter,
            "union": union,
            "jaccard": (inter / union) if union else None,
            "overlap_vs_smaller": (inter / min(len(setA), len(setB))) if min(len(setA), len(setB)) else None,
        },
        "same_id_exact_matches": {
            "matches": same_id,
            "denom": min(len(invA), len(invB)),
            "ratio": same_id / min(len(invA), len(invB)) if min(len(invA), len(invB)) else None,
        },
        "ascii_offset_histogram_top10": [{"offset": o, "count": c} for o, c in top_offsets],
        "best_ascii_offset": best_offset,
        "best_offset_longest_contiguous_run": {
            "run_len": run_len,
            "run_start_in_A": run_start,
            "sample_tokens_A": sampleA,
            "sample_tokens_B": sampleB,
        },
    }


# ----------------------------
# Safetensors remote (Range)
# ----------------------------

DTYPE_BYTES = {"F16":2, "BF16":2, "F32":4, "I32":4, "I64":8, "U8":1}

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
            last_err = RuntimeError(f"HTTP {r.status_code} for {url} range {start}-{end}")
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Range GET failed after retries: {last_err}")

def fetch_safetensors_header(repo_id: str, filename: str, revision: str, token: Optional[str], max_header_bytes: int) -> Dict[str, Any]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = int.from_bytes(raw8, "little")
    if header_len > max_header_bytes:
        raise RuntimeError(
            f"[{repo_id}] header too large for {filename}: {header_len} bytes > max_header_bytes={max_header_bytes}. "
            f"Try increasing --max-header-mb."
        )
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header

def tensor_nbytes(info: Dict[str, Any]) -> int:
    a, b = info["data_offsets"]
    return int(b) - int(a)

def decode_tensor(raw: bytes, dtype: str, shape: List[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = (u16.astype(np.uint32) << 16)
        arr = u32.view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "I32":
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    elif dtype == "I64":
        arr = np.frombuffer(raw, dtype=np.int64).astype(np.float32)
    elif dtype == "U8":
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported dtype: {dtype}")

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


# ----------------------------
# Similarities
# ----------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def align_expert_rows_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """
    Align expert rows (permutation-invariant) and compute mean cosine of matched rows.
    A, B: 2D matrices. We'll coerce to (E, H) by transposing if needed.
    Requires SciPy for optimal; otherwise greedy.
    """
    if A.ndim != 2 or B.ndim != 2:
        return float("nan")

    # convert to (E, H): prefer smaller dim as E if plausible
    def to_EH(X):
        if X.shape[0] <= X.shape[1]:
            return X
        return X.T

    A = to_EH(A).astype(np.float32)
    B = to_EH(B).astype(np.float32)
    E = min(A.shape[0], B.shape[0])
    A = A[:E]
    B = B[:E]

    # normalize rows
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)

    S = A @ B.T  # (E, E)

    if HAVE_SCIPY:
        r, c = linear_sum_assignment(-S)  # maximize S
        return float(S[r, c].mean())

    # greedy fallback
    used = set()
    scores = []
    for i in range(E):
        j = int(np.argmax(S[i]))
        while j in used:
            S[i, j] = -1e9
            j = int(np.argmax(S[i]))
        used.add(j)
        scores.append(float(S[i, j]))
    return float(np.mean(scores))


# ----------------------------
# Key discovery & layer mapping
# ----------------------------

LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")

@dataclasses.dataclass
class TensorKeyRef:
    key: str
    shard: str

def get_layer_index(key: str) -> Optional[int]:
    m = LAYER_RE.search(key)
    if not m:
        return None
    return int(m.group(1))

def classify_key(key: str) -> Optional[str]:
    """
    Heuristic category labels to compare across models.
    We try to pick:
      - norm_pre  (input/pre-attn)
      - norm_post (post-attn / ffn)
      - router    (gate/router)
    """
    lk = key.lower()
    if not lk.endswith(".weight"):
        return None

    # router/gate
    if any(x in lk for x in ["router", "gate", "gating"]) and ("proj" not in lk):
        return "router_or_gate"

    # norms
    if "norm" in lk or "layernorm" in lk or "rms" in lk:
        # exclude final model norm if it contains no layer index; layer index check done elsewhere
        # pre norm
        if any(x in lk for x in ["input", "pre", "attn_norm", "attention_norm", "ln_1", "norm1", "layernorm1"]):
            return "norm_pre"
        # post norm
        if any(x in lk for x in ["post", "ffn", "mlp", "ln_2", "norm2", "layernorm2"]):
            return "norm_post"
        return "norm_any"

    return None

def build_layer_keyrefs(weight_map: Dict[str, str], max_layers: int) -> Dict[int, Dict[str, TensorKeyRef]]:
    """
    For each layer i, pick at most one key per category.
    Priority: norm_pre/norm_post/router_or_gate; fallback norm_any.
    """
    per_layer = defaultdict(lambda: defaultdict(list))
    for k, shard in weight_map.items():
        li = get_layer_index(k)
        if li is None or li >= max_layers:
            continue
        cat = classify_key(k)
        if cat is None:
            continue
        per_layer[li][cat].append(TensorKeyRef(key=k, shard=shard))

    out: Dict[int, Dict[str, TensorKeyRef]] = {}
    for li, cats in per_layer.items():
        chosen: Dict[str, TensorKeyRef] = {}
        # deterministic: sort keys so choice stable
        for cat, lst in cats.items():
            lst_sorted = sorted(lst, key=lambda x: x.key)
            chosen[cat] = lst_sorted[0]

        # if norm_pre missing but norm_any exists, use it as norm_pre fallback
        if "norm_pre" not in chosen and "norm_any" in chosen:
            chosen["norm_pre"] = chosen["norm_any"]
        if "norm_post" not in chosen and "norm_any" in chosen:
            chosen["norm_post"] = chosen["norm_any"]

        out[li] = chosen

    return out

def compute_layer_alignment(
    layersA: List[int],
    layersB: List[int],
    get_vecA,
    get_vecB
) -> Dict[int, int]:
    """
    Align B layers to A layers based on cosine similarity of norm_pre vectors.
    Returns mapping: b_layer -> a_layer.
    Uses Hungarian if SciPy available; otherwise greedy.
    """
    LA = len(layersA)
    LB = len(layersB)
    S = np.zeros((LB, LA), dtype=np.float32)

    for i, lb in enumerate(layersB):
        vb = get_vecB(lb)
        vb = vb.reshape(-1).astype(np.float32)
        nb = np.linalg.norm(vb) + 1e-9
        vb = vb / nb
        for j, la in enumerate(layersA):
            va = get_vecA(la).reshape(-1).astype(np.float32)
            na = np.linalg.norm(va) + 1e-9
            va = va / na
            S[i, j] = float(np.dot(vb, va))

    if HAVE_SCIPY:
        r, c = linear_sum_assignment(-S)  # maximize
        mapping = {layersB[int(rr)]: layersA[int(cc)] for rr, cc in zip(r, c)}
        return mapping

    # greedy fallback
    mapping = {}
    usedA = set()
    for i, lb in enumerate(layersB):
        j = int(np.argmax(S[i]))
        while j in usedA:
            S[i, j] = -1e9
            j = int(np.argmax(S[i]))
        usedA.add(j)
        mapping[lb] = layersA[j]
    return mapping


# ----------------------------
# Main probe
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="upstage/Solar-Open-100B", help="model A repo_id (default Solar)")
    ap.add_argument("--b", default="zai-org/GLM-4.5-Air", help="model B repo_id (default GLM-4.5-Air)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--outdir", default="out_probe")
    ap.add_argument("--max-tensor-bytes", type=int, default=2_500_000, help="skip tensors larger than this")
    ap.add_argument("--max-header-mb", type=int, default=64, help="max safetensors header size to download per shard (MB)")
    ap.add_argument("--align-experts", action="store_true", help="align router/gate experts before scoring (perm-invariant)")
    ap.add_argument("--align-layers", action="store_true", help="align layers B->A based on norm_pre before comparisons")
    ap.add_argument("--max-layers", type=int, default=-1, help="cap number of layers to probe (default: min(num_hidden_layers))")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    ensure_outdir(args.outdir)
    max_header_bytes = args.max_header_mb * 1024 * 1024

    # ---- (1) Config compare
    cfgA = load_config(args.a, args.revision, token)
    cfgB = load_config(args.b, args.revision, token)

    def pick_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "model_type","architectures","hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads",
            "intermediate_size","moe_intermediate_size","n_routed_experts","n_shared_experts","num_experts_per_tok",
            "max_position_embeddings","rope_theta","partial_rotary_factor","first_k_dense_replace",
            "vocab_size","pad_token_id","bos_token_id","eos_token_id","num_nextn_predict_layers",
            "tie_word_embeddings","torch_dtype","transformers_version",
        ]
        out = {}
        for k in keys:
            if k in cfg:
                out[k] = cfg[k]
        return out

    config_report = {
        "A": {"repo": args.a, "config_summary": pick_cfg(cfgA)},
        "B": {"repo": args.b, "config_summary": pick_cfg(cfgB)},
    }
    write_json(os.path.join(args.outdir, "00_config_compare.json"), config_report)

    # ---- (2) Tokenizer compare
    tok_report = tokenizer_compare(args.a, args.b, args.revision, token)
    write_json(os.path.join(args.outdir, "01_tokenizer_compare.json"), tok_report)

    # ---- (3) Weight probe (layerwise)
    # Load index files (required for sharded safetensors)
    try:
        idxA = load_index_json(args.a, args.revision, token)
        idxB = load_index_json(args.b, args.revision, token)
    except Exception as e:
        raise SystemExit(
            f"Failed to download model.safetensors.index.json.\n"
            f"Error: {e}\n"
            f"If the repo truly doesn't have it, this script needs a heavier fallback."
        )

    wmA: Dict[str, str] = idxA.get("weight_map", {})
    wmB: Dict[str, str] = idxB.get("weight_map", {})
    if not wmA or not wmB:
        raise SystemExit("weight_map empty in index.json for one of the models.")

    # Decide layer count to probe
    layersA = int(cfgA.get("num_hidden_layers", 0))
    layersB = int(cfgB.get("num_hidden_layers", 0))
    if layersA <= 0 or layersB <= 0:
        raise SystemExit("Could not read num_hidden_layers from config.json.")

    probe_layers = min(layersA, layersB)
    if args.max_layers > 0:
        probe_layers = min(probe_layers, args.max_layers)

    # Build per-layer candidate tensor keys
    refsA = build_layer_keyrefs(wmA, max_layers=layersA)
    refsB = build_layer_keyrefs(wmB, max_layers=layersB)

    # Determine which layers are present in both
    common_layers = sorted(set(refsA.keys()) & set(refsB.keys()))
    # We will only probe up to probe_layers by default (0..probe_layers-1) but key availability matters.
    common_layers = [L for L in common_layers if L < probe_layers]

    if len(common_layers) < 4:
        eprint("Warning: very few common layers found. Key naming may differ more than expected.")
        eprint("Tip: inspect a few keys from index.json to adjust classify_key().")

    # Caches for headers: (repo, shard)->header
    header_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_header(repo: str, shard: str) -> Dict[str, Any]:
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_safetensors_header(repo, shard, args.revision, token, max_header_bytes=max_header_bytes)
        return header_cache[k]

    # Helper: fetch tensor by key using index->shard and shard header offsets
    def get_tensor_by_key(repo: str, key: str, shard: str) -> np.ndarray:
        header = get_header(repo, shard)
        if key not in header:
            raise KeyError(f"Key not in header: {repo} {shard} {key}")
        return fetch_tensor(repo, shard, args.revision, key, header, token)

    # Optional: layer alignment B->A based on norm_pre vectors
    layer_map: Dict[int, int] = {L: L for L in common_layers}
    if args.align_layers:
        # collect norm_pre vectors for all layers available
        layersA_avail = sorted([L for L in refsA.keys() if "norm_pre" in refsA[L] and L < layersA])
        layersB_avail = sorted([L for L in refsB.keys() if "norm_pre" in refsB[L] and L < layersB])

        # cap to probe_layers
        layersA_avail = [L for L in layersA_avail if L < layersA]
        layersB_avail = [L for L in layersB_avail if L < probe_layers]

        if not layersA_avail or not layersB_avail:
            eprint("align-layers: not enough norm_pre keys found; skipping alignment.")
        else:
            # cache vectors to avoid repeated HTTP
            vecA_cache: Dict[int, np.ndarray] = {}
            vecB_cache: Dict[int, np.ndarray] = {}

            def vecA(la: int) -> np.ndarray:
                if la not in vecA_cache:
                    ref = refsA[la]["norm_pre"]
                    vecA_cache[la] = get_tensor_by_key(args.a, ref.key, ref.shard)
                return vecA_cache[la]

            def vecB(lb: int) -> np.ndarray:
                if lb not in vecB_cache:
                    ref = refsB[lb]["norm_pre"]
                    vecB_cache[lb] = get_tensor_by_key(args.b, ref.key, ref.shard)
                return vecB_cache[lb]

            layer_map = compute_layer_alignment(layersA_avail, layersB_avail, vecA, vecB)
            write_json(os.path.join(args.outdir, "layer_alignment.json"), {
                "note": "mapping is B_layer -> A_layer based on norm_pre cosine similarity (Hungarian if SciPy available else greedy)",
                "have_scipy": HAVE_SCIPY,
                "mapping_B_to_A": {str(k): int(v) for k, v in layer_map.items()},
            })

            # Use aligned layers intersection for probing
            common_layers = sorted([lb for lb in common_layers if lb in layer_map])

    rows: List[Dict[str, Any]] = []
    categories = ["norm_pre", "norm_post", "router_or_gate"]

    for lb in common_layers:
        la = layer_map.get(lb, lb)

        catsA = refsA.get(la, {})
        catsB = refsB.get(lb, {})

        for cat in categories:
            if cat not in catsA or cat not in catsB:
                continue

            refA = catsA[cat]
            refB = catsB[cat]

            try:
                # fetch headers first to know bytes
                hA = get_header(args.a, refA.shard)
                hB = get_header(args.b, refB.shard)
                if refA.key not in hA or refB.key not in hB:
                    continue

                bytesA = tensor_nbytes(hA[refA.key])
                bytesB = tensor_nbytes(hB[refB.key])
                if bytesA > args.max_tensor_bytes or bytesB > args.max_tensor_bytes:
                    continue

                A = fetch_tensor(args.a, refA.shard, args.revision, refA.key, hA, token)
                B = fetch_tensor(args.b, refB.shard, args.revision, refB.key, hB, token)

                # transpose compatibility
                if A.shape != B.shape and A.ndim == 2 and B.ndim == 2 and A.shape == B.shape[::-1]:
                    B = B.T

                if A.shape != B.shape:
                    continue

                aligned_cos = ""
                if cat == "router_or_gate" and args.align_experts:
                    aligned_cos = f"{align_expert_rows_cosine(A, B):.6f}"

                rows.append({
                    "layer_B": lb,
                    "layer_A": la,
                    "category": cat,
                    "shape": str(tuple(A.shape)),
                    "bytes": int(bytesA),
                    "cosine": f"{cosine(A, B):.6f}",
                    "pearson": f"{pearson(A, B):.6f}",
                    "aligned_expert_cosine": aligned_cos,
                    "key_A": refA.key,
                    "key_B": refB.key,
                    "shard_A": refA.shard,
                    "shard_B": refB.shard,
                })
            except Exception as e:
                # keep going; log minimal error
                eprint(f"[warn] layer B={lb} A={la} cat={cat} failed: {e}")

    csv_path = os.path.join(args.outdir, "layerwise_similarity.csv")
    cols = [
        "layer_B","layer_A","category","cosine","pearson","aligned_expert_cosine","bytes","shape",
        "key_A","key_B","shard_A","shard_B"
    ]
    write_csv(csv_path, rows, cols)
    print(f"[ok] wrote {csv_path} (rows={len(rows)})")

    # ---- Plotting
    # aggregate mean per layer_B per category
    per = defaultdict(lambda: defaultdict(list))
    per_p = defaultdict(lambda: defaultdict(list))
    per_aligned = defaultdict(list)

    for r in rows:
        lb = int(r["layer_B"])
        cat = r["category"]
        try:
            per[lb][cat].append(float(r["cosine"]))
            per_p[lb][cat].append(float(r["pearson"]))
        except Exception:
            pass
        if r["category"] == "router_or_gate" and r["aligned_expert_cosine"]:
            try:
                per_aligned[lb].append(float(r["aligned_expert_cosine"]))
            except Exception:
                pass

    layers_sorted = sorted(per.keys())
    cats_present = sorted({r["category"] for r in rows})

    def plot_metric(metric_name: str, store: Dict[int, Dict[str, List[float]]], outname: str):
        plt.figure()
        for cat in cats_present:
            xs = []
            ys = []
            for lb in layers_sorted:
                vals = store[lb].get(cat, [])
                if not vals:
                    continue
                xs.append(lb)
                ys.append(float(np.mean(vals)))
            if len(xs) >= 2:
                plt.plot(xs, ys, marker="o", label=cat)
        plt.title(metric_name)
        plt.xlabel("Layer (B index)")
        plt.ylabel(metric_name.split(":")[-1].strip())
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = os.path.join(args.outdir, outname)
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[ok] wrote {out_png}")

    plot_metric("Layerwise: cosine", per, "plot_cosine_by_layer.png")
    plot_metric("Layerwise: pearson", per_p, "plot_pearson_by_layer.png")

    if args.align_experts and per_aligned:
        xs = []
        ys = []
        for lb in sorted(per_aligned.keys()):
            xs.append(lb)
            ys.append(float(np.mean(per_aligned[lb])))
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.title("Layerwise: router/gate cosine (expert-aligned)")
        plt.xlabel("Layer (B index)")
        plt.ylabel("aligned cosine")
        plt.grid(True, alpha=0.3)
        out_png = os.path.join(args.outdir, "plot_router_aligned_cosine.png")
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[ok] wrote {out_png}")

    # Final note
    print("\n[done]")
    print("- 낮은 상관이 '가중치 재사용 아님'을 곧바로 의미하진 않습니다(continual pretraining, MoE expert permutation 등).")
    print("- 하지만 여러 레이어/여러 텐서에서 비정상적으로 높은 유사도가 반복되면 매우 강한 정황이 됩니다.")
    if args.align_experts and not HAVE_SCIPY:
        print("- (참고) SciPy가 없어서 expert 정렬은 greedy로 수행했습니다. 더 정확히 하려면 pip install scipy")

if __name__ == "__main__":
    main()
