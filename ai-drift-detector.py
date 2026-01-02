# ==============================================================================
# Ghost Drift Audit v10.0 (Global Edition)
# ==============================================================================
# [Concept]
#  Eliminating LightGBM and statistical estimation, this script uses a deterministic
#  "Fejér-Yukawa Kernel" and "FFT Convolution" to mathematically extract data structures.
#
# [Features (All Integrated)]
#  1. Strict Data Binding: Immediately aborts if real data is not found (no fallback to synthetic).
#  2. Source Identity: Records SHA256 hash, size, and row count of the input file in the audit log.
#  3. Logic Identity: Records a sorted logic hash independent of function definition order.
#  4. Tamper-Evident: Verifiable via JSONL hash chain and Strict JSON serialization.
#  5. Cross-Run Chain: Recovers log history from an existing ZIP and appends new records, proving full history with a single file.
#  6. Scale-Invariant UWP: Strict judgment using normalized metrics independent of data magnitude (digits).
#  7. Epsilon Guard: Guard against misidentifying tiny FFT numerical errors as structural negative values.
#  8. BOM Resilience: Normalizes BOM-prefixed CSV column names to fully prevent datetime column misses.
# ==============================================================================

import os
import sys
import platform
import inspect
from typing import Optional, List, Dict, Any, Tuple

# ==============================================================================
# SECTION 0: DETERMINISTIC ENVIRONMENT SETUP
# (MUST BE EXECUTED BEFORE NUMPY IMPORT)
# ==============================================================================
ENV_SNAPSHOT_PRE_NUMPY: Dict[str, Any] = {}

def apply_env(threads: int = 1, pythonhashseed: int = 0) -> None:
    """
    Sets environment variables to force single-threaded execution and fixed hashing.
    CRITICAL: This runs before numpy initialization to strengthen reproducibility.
    """
    global ENV_SNAPSHOT_PRE_NUMPY
    
    if "numpy" in sys.modules:
        print("!! WARNING: 'numpy' is already loaded. Best-effort determinism only in this session. !!")
    
    # Snapshot "pre-numpy" determinism-relevant flags
    hr = int(getattr(sys.flags, "hash_randomization", 1))
    ENV_SNAPSHOT_PRE_NUMPY = {
        "numpy_preloaded_before_apply_env": bool("numpy" in sys.modules),
        "hash_randomization": hr,
        "pythonhashseed_effective": bool(hr == 0),
    }
    
    # Force single-threaded execution for BLAS/MKL/OMP
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    
    # Fix Python's hash seed
    os.environ["PYTHONHASHSEED"] = str(pythonhashseed)

def snapshot_env() -> dict:
    keys = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED"
    ]
    snap = {k: os.environ.get(k, None) for k in keys}
    snap.update(ENV_SNAPSHOT_PRE_NUMPY)
    return snap

# Apply environment immediately
apply_env(threads=1, pythonhashseed=42)

# ==============================================================================
# IMPORTS
# ==============================================================================
import json
import time
import hashlib
import glob
import zipfile
import shutil
import numpy as np
import pandas as pd

# ==============================================================================
# AUDIT CONFIGURATION
# ==============================================================================
AUDIT_CONFIG: Dict[str, Any] = {
    'PROFILE': "commercial",
    'USE_SYNTHETIC_DATA': False, # Strict: Must be False
    'POWER_USAGE_CSV_PATHS': [
        "power_usage.csv",
        "*power_usage*.csv",
        "data/power_usage.csv",
    ],
    'EXTERNAL_CSV_PATHS': [
        "electric_load_weather.csv",
        "jma_weather.csv",
        "demand_weather_merged.csv",
        "target_data.csv",
    ],
    'SYNTHETIC_SEED': 42,
}

# Single-artifact output
AUDIT_BUNDLE_ZIP = "audit_bundle.zip"
PREV_BUNDLE_TMP = "audit_bundle_prev.zip"
AUDIT_RECORD_JSON = "audit_record.json"

# ==============================================================================
# SECTION 1: AUDIT LOG & UTILS
# ==============================================================================

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hash_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return sha256_bytes(x.tobytes())

def hash_file(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def file_size_bytes(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except Exception:
        return -1

# Forward declarations for compute_logic_hash
def fejer_cic2(Gamma: int) -> np.ndarray: pass
def yukawa_fir(lam: float, M: int) -> np.ndarray: pass
def fy_kernel(Gamma: int, lam: float, M: int) -> np.ndarray: pass
def _next_pow2(n: int) -> int: pass
def overlap_save_fft(x: np.ndarray, h: np.ndarray, block: int = 4096) -> np.ndarray: pass
def window_metrics(x: np.ndarray, W: int, eps: float = 1e-9) -> dict: pass
def uwp_pass(metrics: dict, tau_rms: float, tau_neg: float) -> bool: pass
def log_binning(n: np.ndarray, v: np.ndarray, M_bins: int): pass
def cosine_eval(b_smooth: np.ndarray, y_centers: np.ndarray, T: np.ndarray) -> np.ndarray: pass
def tune_params(x: np.ndarray, Gamma_grid, lam_grid, M_grid, W, tau_rms, tau_neg, block=4096, eps=1e-9): pass
def load_data_source(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: pass
def make_audit_record(config: dict, x_in: np.ndarray, y_out: np.ndarray, metrics: dict, prev_record_sha256: Optional[str], outputs: dict) -> dict: pass
def main(): pass

def compute_logic_hash() -> Dict[str, str]:
    """
    Captures the identity of the running code.
    Works for both .py files (File Hash) and Interactive Cells (Logic Hash).
    """
    fpath = globals().get("__file__")
    
    # 1. File Integrity (Strongest)
    if fpath and os.path.exists(fpath):
        return {
            "type": "file",
            "path": os.path.basename(fpath), # Exclude full path for portability
            "sha256": hash_file(fpath)
        }
    
    # 2. Logic Integrity (Fallback for Notebooks)
    # Hashes the source code of critical functions to guarantee logic version.
    try:
        # Collect objects
        objs = [
            apply_env,
            fejer_cic2, yukawa_fir, fy_kernel,
            _next_pow2, overlap_save_fft,
            window_metrics, uwp_pass,
            log_binning, cosine_eval, tune_params,
            load_data_source,
            make_audit_record,
            main,
        ]
        # Sort by name to ensure deterministic order regardless of definition order
        objs.sort(key=lambda x: x.__name__)
        
        src = ""
        for obj in objs:
            src += inspect.getsource(obj)
        
        return {
            "type": "logic_proxy",
            "note": "Interactive session detected. Hashed sorted core function sources.",
            "sha256": sha256_bytes(src.encode("utf-8"))
        }
    except Exception as e:
        return {
            "type": "unknown",
            "error": str(e),
            "sha256": "missing"
        }

def last_record_hash(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        obj = json.loads(last)
        return obj.get("record_sha256", None)
    except Exception:
        return None

def last_record_hash_from_zip(zip_path: str, member: str = "audit_log.jsonl") -> Optional[str]:
    if not (zip_path and os.path.exists(zip_path)):
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            if member not in z.namelist():
                return None
            with z.open(member, "r") as f:
                last = None
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        last = line
                if not last:
                    return None
                obj = json.loads(last)
                return obj.get("record_sha256", None)
    except Exception:
        return None

def get_prev_record_hash() -> Optional[str]:
    # 1) Prefer local audit_log.jsonl if it exists (continuation in same dir)
    h = last_record_hash("audit_log.jsonl")
    if h:
        return h
    # 2) Otherwise, check the preserved previous bundle
    h = last_record_hash_from_zip(PREV_BUNDLE_TMP, member="audit_log.jsonl")
    if h:
        return h
    return None

def sha256_json(obj: dict) -> str:
    # STRICT serialization for hash consistency
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode("utf-8")
    return sha256_bytes(b)

def append_jsonl(path: str, record: dict) -> None:
    # STRICT serialization
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def bundle_zip(zip_path: str, files: List[str]) -> str:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p and os.path.exists(p):
                # Guard against recursive zip inclusion
                if os.path.abspath(p) == os.path.abspath(zip_path):
                    continue
                z.write(p, arcname=os.path.basename(p))
    return zip_path

def sparkline(x: np.ndarray, width: int = 96) -> str:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0: return ""
    if x.size > width:
        idx = np.linspace(0, x.size - 1, width).astype(int)
        x = x[idx]
    mn, mx = float(np.min(x)), float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return "▁" * int(x.size)
    levels = "▂▃▄▅▆▇█"
    y = (x - mn) / (mx - mn)
    k = np.clip((y * (len(levels) - 1)).astype(int), 0, len(levels) - 1)
    return "".join(levels[i] for i in k)

def make_audit_record(config: dict, 
                      x_in: np.ndarray, 
                      y_out: np.ndarray, 
                      metrics: dict,
                      prev_record_sha256: Optional[str],
                      outputs: dict) -> dict:
    
    core = {
        "mode": config.get("mode"),
        "source_fingerprint": config.get("source_fingerprint"),
        "script_identity_sha256": (config.get("script_identity") or {}).get("sha256"),
        "params": (config.get("params") or {}),
        "M_bins": config.get("M_bins"),
        "W": config.get("W"),
        "tau_rms": config.get("tau_rms"),
        "tau_neg": config.get("tau_neg"),
        "uwp_eps": config.get("uwp_eps"),
        "bins_norm": config.get("bins_norm"),
        "input_sha256": hash_array(x_in),
        "output_sha256": hash_array(y_out),
        "metrics": metrics,
        "outputs": outputs,
    }
    core_sha256 = sha256_json(core)

    rec = {
        "run_id": core_sha256,
        "core_sha256": core_sha256,
        "ts_unix": time.time(),
        "config": config,
        "prev_record_sha256": prev_record_sha256,
        "input_sha256": hash_array(x_in),
        "output_sha256": hash_array(y_out),
        "outputs": outputs,
        "metrics": metrics,
    }
    rec["record_sha256"] = sha256_json(rec)
    return rec

# ==============================================================================
# SECTION 2: KERNEL GENERATION
# ==============================================================================

def fejer_cic2(Gamma: int) -> np.ndarray:
    if Gamma < 0: raise ValueError(f"Gamma must be non-negative: {Gamma}")
    L = 2 * Gamma + 1
    rect = np.ones(L, dtype=np.float64) / L
    tri = np.convolve(rect, rect, mode="full")
    tri = np.maximum(tri, 0.0)
    return tri / tri.sum()

def yukawa_fir(lam: float, M: int) -> np.ndarray:
    if lam <= 0: raise ValueError(f"Lambda must be positive: {lam}")
    n = np.arange(-M, M + 1, dtype=np.int64)
    k = np.exp(-lam * np.abs(n)).astype(np.float64)
    k = np.maximum(k, 0.0)
    return k / k.sum()

def fy_kernel(Gamma: int, lam: float, M: int) -> np.ndarray:
    f = fejer_cic2(Gamma)
    y = yukawa_fir(lam, M)
    k = np.convolve(f, y, mode="full")
    k = np.maximum(k, 0.0)
    return k / k.sum()

# ==============================================================================
# SECTION 3: ROBUST FFT CONVOLUTION
# ==============================================================================

def _next_pow2(n: int) -> int:
    p = 1
    while p < n: p <<= 1
    return p

def overlap_save_fft(x: np.ndarray, h: np.ndarray, block: int = 4096) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if x.size == 0: return x.copy()
    
    Lh = int(h.size)
    if block <= Lh: block = _next_pow2(Lh + 1)
    Nfft = _next_pow2(block + Lh - 1)
    H = np.fft.rfft(h, n=Nfft)
    pad = Lh - 1
    xpad = np.concatenate([np.zeros(pad, dtype=np.float64), x])
    
    out = []
    step = block
    for i in range(0, len(x), step):
        chunk = xpad[i : i + pad + block]
        if len(chunk) < pad + block:
            chunk = np.pad(chunk, (0, pad + block - len(chunk)))
        Y = np.fft.rfft(chunk, n=Nfft) * H
        y_block = np.fft.irfft(Y, n=Nfft)
        out.append(y_block[pad : pad + block])
        
    return np.concatenate(out)[: len(x)]

# ==============================================================================
# SECTION 4: UWP METRICS (EPSILON GUARDED)
# ==============================================================================

def window_metrics(x: np.ndarray, W: int, eps: float = 1e-9) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0: return {"rms_worst": 0.0, "rneg_worst": 0.0, "W": W, "eps": float(eps)}
    if W > len(x): W = len(x)

    # 1. Zero out tiny numerical noise
    if eps > 0.0:
        x = np.where(np.abs(x) < eps, 0.0, x)

    # 2. O(N) scan via prefix sums
    xs = np.concatenate(([0.0], np.cumsum(x)))
    x2s = np.concatenate(([0.0], np.cumsum(x * x)))
    # Negative count: strict inequality below -eps
    neg = (x < -eps).astype(np.float64)
    ns = np.concatenate(([0.0], np.cumsum(neg)))

    m = len(x) - W + 1
    sum_w = xs[W:] - xs[:-W]
    sum2_w = x2s[W:] - x2s[:-W]
    neg_w = ns[W:] - ns[:-W]

    mean = sum_w / W
    ex2 = sum2_w / W
    var = np.maximum(ex2 - mean * mean, 0.0)
    
    return {
        "rms_worst": float(np.max(np.sqrt(var))),
        "rneg_worst": float(np.max(neg_w / W)),
        "W": int(W),
        "eps": float(eps)
    }

def uwp_pass(metrics: dict, tau_rms: float, tau_neg: float) -> bool:
    return (metrics["rms_worst"] <= float(tau_rms)) and (metrics["rneg_worst"] <= float(tau_neg))

# ==============================================================================
# SECTION 5: PRIME PREPROCESSING
# ==============================================================================

def log_binning(n: np.ndarray, v: np.ndarray, M_bins: int):
    n = np.asarray(n, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if np.any(n <= 0.0): n = n - n.min() + 1.0
    
    y = np.log(n)
    y_min, y_max = float(y.min()), float(y.max())
    edges = np.linspace(y_min, y_max, M_bins + 1)
    idx = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, M_bins - 1)
    
    bins = np.zeros(M_bins, dtype=np.float64)
    np.add.at(bins, idx, v)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return bins, centers

def cosine_eval(b_smooth: np.ndarray, y_centers: np.ndarray, T: np.ndarray) -> np.ndarray:
    out = np.empty(T.shape[0], dtype=np.float64)
    chunk = 256
    for i in range(0, T.shape[0], chunk):
        Ti = T[i:i+chunk]
        Ci = np.cos(Ti[:, None] * y_centers[None, :])
        out[i:i+chunk] = Ci @ b_smooth
    return out

def tune_params(x: np.ndarray, Gamma_grid, lam_grid, M_grid, W, tau_rms, tau_neg, block=4096, eps=1e-9):
    best = None
    for Gamma in Gamma_grid:
        for lam in lam_grid:
            for M in M_grid:
                try: k = fy_kernel(int(Gamma), float(lam), int(M))
                except ValueError: continue
                
                Lh = int(k.size)
                blk = _next_pow2(max(block, Lh + 1))
                Nfft = _next_pow2(blk + Lh - 1)
                cost_proxy = float(((len(x)+blk)/blk) * Nfft * np.log2(max(Nfft, 2)) + Lh)
                
                # FIXED: Use blk instead of default block for execution
                y = overlap_save_fft(x, k, block=blk)
                # UPDATED: Pass eps to window_metrics
                met = window_metrics(y, W=W, eps=eps)
                
                if uwp_pass(met, tau_rms, tau_neg):
                    cand = (cost_proxy, int(Gamma), float(lam), int(M), y, met)
                    if best is None or cand[0] < best[0]:
                        best = cand
    return best

# ==============================================================================
# SECTION 7: ROBUST DATA LOADING (STRICT MODE)
# ==============================================================================

def load_data_source(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads data with Strict Identity checks.
    Raises RuntimeError if NO valid CSV is found (Strict Mode).
    """
    print("[0/6] Loading Data Source (Strict Mode)...")
    
    if config.get("USE_SYNTHETIC_DATA", False):
        print("  -> USE_SYNTHETIC_DATA is True. Generating deterministic synthetic data.")
        rng = np.random.default_rng(config.get('SYNTHETIC_SEED', 42))
        n = np.arange(1, 10000, dtype=np.float64)
        v = rng.normal(loc=1.0, scale=0.5, size=len(n)) + 0.5 * np.sin(n / 50.0)
        return n, v, {"type": "synthetic", "seed": config.get('SYNTHETIC_SEED')}

    # Priority: POWER_USAGE > EXTERNAL
    candidates: List[str] = []
    patterns = list(config.get("POWER_USAGE_CSV_PATHS", [])) + list(config.get("EXTERNAL_CSV_PATHS", []))
    search_dirs = [os.getcwd(), "/mnt/data", "/content"]
    seen = set()
    
    for ptn in patterns:
        if os.path.isabs(ptn):
            hits = sorted(glob.glob(ptn))
        else:
            hits = []
            for d in search_dirs:
                hits.extend(glob.glob(os.path.join(d, ptn)))
            hits = sorted(list(set(hits)))
            
        for hit in hits:
            if hit not in seen and os.path.isfile(hit):
                candidates.append(hit)
                seen.add(hit)

    print(f"  -> Found {len(candidates)} candidates: {[os.path.basename(c) for c in candidates]}")

    for path in candidates:
        try:
            df = None
            # Prioritize utf-8-sig to handle BOM correctly from the start
            for enc in ["utf-8-sig", "utf-8", "shift_jis", "cp932"]:
                try:
                    df_temp = pd.read_csv(path, encoding=enc)
                    # Aggressively clean column names (remove BOM artifacts if utf-8-sig didn't catch it)
                    df_temp.columns = [str(c).replace('\ufeff', '').strip() for c in df_temp.columns]
                    df = df_temp
                    break
                except Exception: continue
            
            if df is None: continue

            # Heuristic Date/Time
            if "DATETIME" not in df.columns:
                if "DATE" in df.columns and "TIME" in df.columns:
                    try: df["DATETIME"] = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str), errors="coerce")
                    except: pass
                elif "年月日時" in df.columns:
                    df["DATETIME"] = pd.to_datetime(df["年月日時"], errors="coerce")
                elif "日付" in df.columns and "時刻" in df.columns:
                     try: df["DATETIME"] = pd.to_datetime(df["日付"].astype(str) + " " + df["時刻"].astype(str), errors="coerce")
                     except: pass

            if "DATETIME" in df.columns:
                df = df.dropna(subset=["DATETIME"]).sort_values("DATETIME").reset_index(drop=True)
            else:
                df = df.reset_index(drop=True)

            # Value Column (strong preference for power usage datasets)
            val_col = None
            if any(k in os.path.basename(path).lower() for k in ["power_usage", "power-usage", "powerusage"]):
                for c in ["Power usage", "power usage", "usage", "load", "demand"]:
                    match = next((col for col in df.columns if c.lower() == str(col).strip().lower()), None)
                    if match:
                        val_col = match
                        break
            
            if val_col is None:
                for c in ["当日実績(万kW)", "需要", "電力需要", "DEMAND", "TEMP", "value", "y"]:
                    match = next((col for col in df.columns if c in str(col)), None)
                    if match:
                        val_col = match
                        break
            
            if val_col is None:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols: val_col = num_cols[0]

            if val_col is None: continue

            v = df[val_col].values.astype(np.float64)
            n = np.arange(1, len(v) + 1, dtype=np.float64)
            
            file_hash = hash_file(path)
            meta = {
                "type": "csv",
                "filename": os.path.basename(path),
                "sha256": file_hash,
                "bytes": file_size_bytes(path),
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "target_col": val_col,
                "has_datetime": "DATETIME" in df.columns
            }
            print(f"  -> LOCKED ON: {os.path.basename(path)} ({val_col})")
            return n, v, meta

        except Exception as e:
            print(f"  -> Error reading {os.path.basename(path)}: {e}")
            continue

    raise RuntimeError(
        "FATAL: No valid CSV found in configured paths. "
        "Strict Audit Mode forbids fallback to synthetic data."
    )

# ==============================================================================
# SECTION 8: MAIN EXECUTION
# ==============================================================================

def main():
    print("--- FY-UWP Ghost Drift Audit v10.0 (Global Edition) ---")
    
    # 0. Chain Preservation (Recover previous logs before overwriting)
    try:
        if os.path.exists(PREV_BUNDLE_TMP):
            os.remove(PREV_BUNDLE_TMP)
        if os.path.exists(AUDIT_BUNDLE_ZIP):
            print(f"[Init] Preserving existing bundle '{AUDIT_BUNDLE_ZIP}' for history chain...")
            with zipfile.ZipFile(AUDIT_BUNDLE_ZIP, "r") as z:
                if "audit_log.jsonl" in z.namelist():
                    z.extract("audit_log.jsonl", ".")
            shutil.move(AUDIT_BUNDLE_ZIP, PREV_BUNDLE_TMP)
    except Exception as e:
        print(f"[Init] Warning during bundle preservation: {e}")

    # 1. Load Data
    try:
        n, v, source_meta = load_data_source(AUDIT_CONFIG)
    except RuntimeError as e:
        print(f"\n[!] ABORT: {e}")
        return

    print(f"[1/6] Data Loaded: {len(n)} points. Source: {source_meta.get('filename', 'synthetic')}")
    
    # 2. Log-Binning & Scale Normalization
    print("[2/6] Logarithmic binning & Scale Normalization...")
    M_bins = min(2048, len(n) // 2) if len(n) > 100 else 64
    bins, centers = log_binning(n, v, M_bins=M_bins)
    
    # [FEATURE] Scale-Invariant Normalization
    EPS_UWP = 1e-9
    bins_raw = bins.copy() # Keep raw for output hashing
    rms_scale = float(np.sqrt(np.mean(bins_raw * bins_raw))) if bins_raw.size > 0 else 1.0
    bins = bins_raw / (rms_scale + EPS_UWP)
    
    # 3. Setup & Tune
    print("[3/6] Kernel Application & Tuning...")
    Gamma0, lam0, M0 = 5, 0.6, 256
    W, tau_rms, tau_neg = 128, 0.15, 0.40
    
    try:
        k = fy_kernel(Gamma0, lam0, M0)
        b_smooth = overlap_save_fft(bins, k, block=4096)
        # [FEATURE] Epsilon Guard
        met = window_metrics(b_smooth, W=W, eps=EPS_UWP)
    except Exception:
        met = {"rms_worst": 999.9, "rneg_worst": 999.9, "W": int(W), "eps": float(EPS_UWP)}

    passed = uwp_pass(met, tau_rms, tau_neg)
    if not passed:
        print("      Constraint violation. Tuning...")
        # Tune with epsilon awareness
        best = tune_params(bins, [2, 5, 10], [0.4, 0.6, 0.8], [128, 256, 512], W, tau_rms, tau_neg, eps=EPS_UWP)
        if best:
            _, Gamma0, lam0, M0, b_smooth, met = best
            passed = uwp_pass(met, tau_rms, tau_neg)
            print(f"      Optimized: Gamma={Gamma0}, lam={lam0}, M={M0}")
        else:
            print("      WARNING: Optimization failed.")
            if 'b_smooth' not in locals():
                b_smooth = bins
            passed = False
    
    # Stamp verdict into metrics
    met = dict(met)
    met["uwp_pass"] = bool(passed)
    met["tau_rms"] = float(tau_rms)
    met["tau_neg"] = float(tau_neg)

    # 4. Final Eval
    print("[5/6] Calculating S_FY(t)...")
    T = np.linspace(0, 50, 200)
    S = cosine_eval(b_smooth, centers, T)
    
    # 5. Audit Logging
    print("[6/6] Finalizing Bundle...")
    
    # Compute Script Identity (Sorted logic hash)
    script_info = compute_logic_hash()
    
    # Portable fingerprint
    source_fingerprint = {
        "type": source_meta.get("type"),
        "filename": source_meta.get("filename"),
        "sha256": source_meta.get("sha256"),
        "bytes": source_meta.get("bytes", None),
        "rows": source_meta.get("rows"),
        "cols": source_meta.get("cols"),
        "target_col": source_meta.get("target_col"),
        "has_datetime": source_meta.get("has_datetime"),
    }
    
    config = {
        "mode": "fy_uwp_strict_norm",
        "source_meta": source_meta,
        "source_fingerprint": source_fingerprint,
        "params": {"Gamma": Gamma0, "lambda": lam0, "M": M0},
        "Gamma": int(Gamma0),
        "lambda": float(lam0),
        "M": int(M0),
        "M_bins": int(M_bins),
        "W": int(W),
        "tau_rms": float(tau_rms),
        "tau_neg": float(tau_neg),
        "uwp_eps": float(EPS_UWP),
        "bins_norm": {"method": "rms", "scale": float(rms_scale), "eps": float(EPS_UWP)},
        "env": snapshot_env(),
        "script_identity": script_info,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cwd": os.path.basename(os.getcwd()),
        "argv": [os.path.basename(str(a)) for a in sys.argv],
    }
    
    outputs = {
        "bins_raw_sha256": hash_array(bins_raw), # Hash raw data
        "S_array_sha256": hash_array(S),
        "b_smooth_array_sha256": hash_array(b_smooth),
        "T_array_sha256": hash_array(T),
        "centers_array_sha256": hash_array(centers),
    }
    
    prev = get_prev_record_hash()
    rec = make_audit_record(config, bins, b_smooth, met, prev, outputs)
    append_jsonl("audit_log.jsonl", rec)

    # In-thread record
    print("\n--- BEGIN_GITHUB_RECORD ---")
    print("```text")
    print(f"Ghost Drift Audit v10.0 (Global Edition)")
    print(f"RunID(core): {rec['core_sha256']}")
    print(f"RecordSHA256: {rec['record_sha256']}")
    print(f"PrevSHA256: {rec.get('prev_record_sha256')}")
    print(f"Source: {source_fingerprint.get('filename')} (SHA256: {source_fingerprint.get('sha256')[:8]}...)")
    print(f"LogicID: {script_info.get('sha256')} ({script_info.get('type')})")
    print(f"Norm: method=rms, scale={rms_scale:.4g}, eps={EPS_UWP}")
    print(f"UWP: pass={met.get('uwp_pass')}, rms={met.get('rms_worst'):.4f}, neg={met.get('rneg_worst'):.4f} (tau={met.get('tau_rms')})")
    print(f"S_sparkline: {sparkline(S)}")
    print("```")
    print("--- END_GITHUB_RECORD ---\n")
    
    # Certificate (Strict JSON)
    cert = {
        "run_id": rec["run_id"],
        "core_sha256": rec["core_sha256"],
        "record_sha256": rec["record_sha256"],
        "prev_record_sha256": rec.get("prev_record_sha256"),
        "timestamp": rec["ts_unix"],
        "source": source_meta,
        "script_identity": script_info,
        "metrics": met,
        "config": config,
        "notes": "Strict Audit Mode: Logic Sealed + Scale Norm + Chain + BOM Fix."
    }
    write_text(AUDIT_RECORD_JSON, json.dumps(cert, ensure_ascii=False, sort_keys=True, indent=2))
    
    # Bundle
    files_to_bundle = ["audit_log.jsonl", AUDIT_RECORD_JSON]
    bundle_zip(AUDIT_BUNDLE_ZIP, files_to_bundle)
    
    # Cleanup
    try:
        if os.path.exists("audit_log.jsonl"): os.remove("audit_log.jsonl")
        if os.path.exists(AUDIT_RECORD_JSON): os.remove(AUDIT_RECORD_JSON)
        if os.path.exists(PREV_BUNDLE_TMP): os.remove(PREV_BUNDLE_TMP)
    except: pass
    
    print(f"DONE. Generated: {AUDIT_BUNDLE_ZIP}")

if __name__ == "__main__":
    main()
