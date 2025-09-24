# app/model_runner.py
import os, shlex, subprocess, tempfile
from pathlib import Path

REPO_ROOT = Path("/app/index-tts")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/index-tts/checkpoints"))
INFER_MODULE = os.getenv("INFER_MODULE", "indextts.infer_v2")  # v2 entry

def _require(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def load() -> None:
    """Validate repo + checkpoints exist."""
    _require(REPO_ROOT / "indextts" / "infer_v2.py", "indextts/infer_v2.py")
    _require(MODEL_DIR, "MODEL_DIR (checkpoints directory)")
    _require(MODEL_DIR / "config.yaml", "config.yaml in checkpoints")
    # Heuristic heads-up
    maybe = [MODEL_DIR / "gpt.pth", MODEL_DIR / "s2mel.pth"]
    if not all(p.exists() for p in maybe):
        print("[model_runner] Heads-up: gpt.pth / s2mel.pth not both found; continuing…", flush=True)

# app/model_runner.py (add near top)
def _default_ref() -> str | None:
    for p in [
        REPO_ROOT / "examples" / "voice_01.wav",
        REPO_ROOT / "examples" / "voice_02.wav",
        REPO_ROOT / "examples" / "voice_03.wav",
    ]:
        if p.exists():
            return str(p)
    return None

def _build_cli(text: str, out_wav: str, ref_audio: str | None) -> list[str]:
    args = [
        "python3", "-m", INFER_MODULE,
        text,
        "--model_dir", str(MODEL_DIR),
        "--config", str(MODEL_DIR / "config.yaml"),
        "--output", out_wav,
    ]
    ref = ref_audio or _default_ref()  # <- fallback to examples/voice_01.wav if none given
    if ref:
        args += ["--ref", ref]
    else:
        print("[model_runner] No ref provided and no default ref found; synthesis may fail.", flush=True)

    print("[INDEXTTS CMD]", shlex.join(args), flush=True)
    return args

def synthesize_to_wav_bytes(
    text: str,
    ref_audio_path: str | None = None,
    emotion: str | None = None,              # unused by CLI; kept for API compat
    target_duration_s: float | None = None,  # unused by CLI
) -> bytes:
    load()
    with tempfile.TemporaryDirectory() as td:
        out_wav = str(Path(td) / "out.wav")
        cmd = _build_cli(text, out_wav, ref_audio_path)
        # 👇 IMPORTANT: run from the repo root so "checkpoints/..." exists
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        return Path(out_wav).read_bytes()