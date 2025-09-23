import os, io
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download

# Where to fetch and cache the model (set in RunPod "Environment" later)
MODEL_REPO = os.getenv("HF_MODEL_REPO", "IndexTeam/IndexTTS-2")
HF_TOKEN   = os.getenv("HF_TOKEN")  # only if the repo requires it
MODEL_DIR  = os.getenv("MODEL_DIR", "/models/indextts2")

def _ensure_model_downloaded():
    """Download weights once into persistent storage (serverless caches layer or your volume)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    # If empty, pull snapshot. local_dir_use_symlinks=False prevents broken paths in serverless.
    if not any(os.scandir(MODEL_DIR)):
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )

_loaded = False

def load():
    """Called once on cold start. Put real model init here."""
    global _loaded
    if _loaded:
        return
    _ensure_model_downloaded()
    # TODO: import your real IndexTTS-2 code and load to GPU here, e.g.:
    # from indextts2.infer import load_model
    # global _model
    # _model = load_model(MODEL_DIR, device="cuda")
    _loaded = True

def synthesize_to_wav_bytes(
    text: str,
    ref_audio_path: str | None = None,
    emotion: str | None = None,
    target_duration_s: float | None = None,
) -> bytes:
    """
    Replace this placeholder with the real IndexTTS-2 inference call.
    Must return raw WAV bytes.
    """
    # ---- PLACEHOLDER so the endpoint is testable now: 1 second of silence ----
    sr = 22050
    samples = np.zeros(int(sr * 1.0), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()
