import os, io, wave, struct
import numpy as np
from huggingface_hub import snapshot_download

MODEL_REPO = os.getenv("HF_MODEL_REPO", "IndexTeam/IndexTTS-2")
HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_DIR  = os.getenv("MODEL_DIR", "/models/indextts2")

def _ensure_model_downloaded():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # optional: skip download for now; uncomment when you wire the real model
    # if not any(os.scandir(MODEL_DIR)):
    #     snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR,
    #                       token=HF_TOKEN, local_dir_use_symlinks=False)

_loaded = False
def load():
    global _loaded
    if _loaded: return
    _ensure_model_downloaded()
    # TODO: init real IndexTTS-2 here later
    _loaded = True

def _wav_bytes_from_float32(mono: np.ndarray, sr: int = 22050) -> bytes:
    """Write 16-bit PCM WAV (pure stdlib)."""
    # clip & scale to int16
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def synthesize_to_wav_bytes(text, ref_audio_path=None, emotion=None, target_duration_s=None) -> bytes:
    # placeholder so endpoint runs: 1s of silence
    sr = 22050
    samples = np.zeros(sr, dtype=np.float32)
    return _wav_bytes_from_float32(samples, sr)
