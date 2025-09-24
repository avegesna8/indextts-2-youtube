# app/model_runner.py
import os, sys, json, subprocess, tempfile, hashlib
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/app/index-tts")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/index-tts/checkpoints"))
FALLBACK_REF = Path("/app/app/assets/ref.wav")

# --- globals for caching ---
_TTS = None              # singleton model
_LOADED = False          # load() one-time
_CLEAN_CACHE_DIR = Path("/tmp/ref_clean_cache")
_CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _log(msg: str, **kv):
    print("[model_runner]", json.dumps({"msg": msg, **kv}, ensure_ascii=False), flush=True)

def _require(path: Path, desc: str):
    if not path.exists():
        _log("MISSING_PATH", desc=desc, path=str(path))
        raise FileNotFoundError(f"Missing {desc}: {path}")

def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

def _ffmpeg_ok() -> bool:
    return _which("ffmpeg") is not None

def _default_ref() -> Optional[str]:
    return str(FALLBACK_REF) if FALLBACK_REF.exists() else None

def load() -> None:
    """Heavy checks: run once per container."""
    global _LOADED
    if _LOADED:
        return
    _log("ENV", PYTHON=sys.version, CWD=os.getcwd(), REPO_ROOT=str(REPO_ROOT), MODEL_DIR=str(MODEL_DIR))
    _require(REPO_ROOT / "indextts" / "infer_v2.py", "indextts/infer_v2.py")
    _require(MODEL_DIR, "MODEL_DIR (checkpoints directory)")
    _require(MODEL_DIR / "config.yaml", "config.yaml in checkpoints")
    for fname in ["bpe.model", "gpt.pth", "s2mel.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"]:
        _require(MODEL_DIR / fname, fname)
    if not _ffmpeg_ok():
        _log("WARN", detail="ffmpeg not in PATH; ref cleaning will fail if needed")
    
    # --- CUDA info log here ---
    import torch
    _log("CUDA_INFO",
         cuda_available=torch.cuda.is_available(),
         device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
         
    _LOADED = True
    _log("LOAD_OK")

# ---------- Caching helpers ----------
def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _to_clean_wav_cached(src_path: str) -> str:
    """
    Convert arbitrary audio -> PCM16/22.05k/mono and cache by file hash.
    """
    if not _ffmpeg_ok():
        return src_path  # fallback (not ideal but avoids crash)

    key = _hash_file(src_path) + ".wav"
    dst = _CLEAN_CACHE_DIR / key
    if dst.exists():
        return str(dst)

    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-i", src_path, "-ac","1","-ar","22050","-sample_fmt","s16", str(dst)]
    subprocess.run(cmd, check=True)
    return str(dst)

# ---------- Model singleton ----------
from indextts.infer_v2 import IndexTTS2

def _get_tts() -> IndexTTS2:
    global _TTS
    if _TTS is None:
        load()
        _TTS = IndexTTS2(
            cfg_path=str(MODEL_DIR / "config.yaml"),
            model_dir=str(MODEL_DIR),
            use_fp16=True,          # set True if you have GPU + fp16 weights
            use_cuda_kernel=True,   # set True if GPU kernels available
            use_deepspeed=False,
        )
        _log("TTS_INIT_OK")
    return _TTS

def synthesize_to_wav_bytes_api(
    text: str,
    ref_audio_path: Optional[str] = None,
    emotion: Optional[str] = None,
    target_duration_s: Optional[float] = None,
) -> bytes:
    tts = _get_tts()  # <- reuses singleton instance

    # choose & cache-clean the reference once, if provided
    ref = ref_audio_path or _default_ref()
    if ref:
        ref = _to_clean_wav_cached(ref)

    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "gen.wav")
        tts.infer(
            spk_audio_prompt=ref,
            text=text,
            output_path=out_path,
            verbose=False,
        )
        return Path(out_path).read_bytes()
