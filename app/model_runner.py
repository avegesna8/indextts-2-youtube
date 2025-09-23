# app/model_runner.py
import os, io, wave, shlex, subprocess, tempfile
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download

# Weights from HF
MODEL_REPO = os.getenv("HF_MODEL_REPO", "IndexTeam/IndexTTS-2")
HF_TOKEN   = os.getenv("HF_TOKEN")
MODEL_DIR  = os.getenv("MODEL_DIR", "/models/indextts2")

# Path to the GitHub repo we cloned in the Dockerfile
REPO_ROOT  = Path("/app/index-tts")
INFER_PY   = REPO_ROOT / "infer_v2.py"    # recent script in that repo
ALT_INFER  = REPO_ROOT / "tools" / "infer.py"  # fallback if needed

def _ensure_weights():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    # Pull once; HF snapshot is cached across cold starts
    if not any(Path(MODEL_DIR).iterdir()):
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )

_loaded = False
def load():
    """Cold start initialization."""
    global _loaded
    if _loaded:
        return
    _ensure_weights()
    _loaded = True

def _wav_bytes_from_float32(mono: np.ndarray, sr: int = 22050) -> bytes:
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def _build_cli(text: str, out_wav: str, ref_audio: str | None, emotion: str | None, duration: float | None):
    """
    Compose a best-guess command for IndexTTS-2 based on common flags.
    If it errors, check the container logs for the exact flags and tweak here.
    """
    script = INFER_PY if INFER_PY.exists() else ALT_INFER
    # Common flags (adjust if the repo uses different names):
    # --text, --output, --config, --bpe, --gpt, --s2mel, --feat1, --feat2, --ref, --emotion, --duration
    cfg = Path(MODEL_DIR) / "config.yaml"
    bpe = Path(MODEL_DIR) / "bpe.model"
    gpt = Path(MODEL_DIR) / "gpt.pth"
    s2m = Path(MODEL_DIR) / "s2mel.pth"
    f1  = Path(MODEL_DIR) / "feat1.pt"
    f2  = Path(MODEL_DIR) / "feat2.pt"

    args = [
        "python3", str(script),
        "--text", text,
        "--output", out_wav,
        "--config", str(cfg),
        "--bpe", str(bpe),
        "--gpt", str(gpt),
        "--s2mel", str(s2m),
        "--feat1", str(f1),
        "--feat2", str(f2),
    ]
    if ref_audio:
        args += ["--ref", ref_audio]
    if emotion:
        args += ["--emotion", emotion]
    if duration:
        args += ["--duration", str(duration)]
    return args

def synthesize_to_wav_bytes(
    text: str,
    ref_audio_path: str | None = None,
    emotion: str | None = None,
    target_duration_s: float | None = None,
) -> bytes:
    if not _loaded:
        load()

    # Try CLI (reliable since the HF repo shipped only weights)
    with tempfile.TemporaryDirectory() as td:
        out_wav = str(Path(td) / "out.wav")
        cmd = _build_cli(text, out_wav, ref_audio_path, emotion, target_duration_s)
        # For debugging: print("CMD:", shlex.join(cmd))
        subprocess.run(cmd, check=True)
        with open(out_wav, "rb") as f:
            return f.read()
