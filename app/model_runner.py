# app/model_runner.py
# Uses the repo's CLI:  python -m indextts.infer "<TEXT>" --voice <ref.wav> --model_dir ... --config ... --output out.wav
import os
import subprocess
import tempfile
import shlex
from pathlib import Path

# Where you vendored the repo and weights
REPO_ROOT = Path("/app/index-tts")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/index-tts/checkpoints"))

# Optional: override which module to call (default from README)
INFER_MODULE = os.getenv("INFER_MODULE", "indextts.infer")

def _require(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")

def load() -> None:
    """Cold start validation: make sure the repo & checkpoints exist."""
    _require(REPO_ROOT / "indextts" / "infer.py", "indextts/infer.py")
    _require(MODEL_DIR, "MODEL_DIR (checkpoints directory)")
    _require(MODEL_DIR / "config.yaml", "config.yaml in checkpoints")
    # Many repos also need bpe/vocab/weights; fail fast if obviously missing
    expected_any = [
        MODEL_DIR / "gpt.pth",
        MODEL_DIR / "bigvgan_generator.pth",
        MODEL_DIR / "dvae.pth",
    ]
    if not any(p.exists() for p in expected_any):
        # Not fatal if you have alternative filenames, but warn loudly
        print("[model_runner] Heads-up: did not see gpt/bigvgan/dvae .pth files in checkpoints. "
              "If your clone uses different names, that's OK.", flush=True)

def _build_cli(text: str, out_wav: str, ref_audio: str | None) -> list[str]:
    """Build the exact CLI the README documents."""
    args = [
        "python3", "-m", INFER_MODULE,
        text,  # positional (not --text)
        "--model_dir", str(MODEL_DIR),
        "--config", str(MODEL_DIR / "config.yaml"),
        "--output", out_wav,
    ]
    if ref_audio:
        args += ["--voice", ref_audio]
    # Log the command for debugging
    print("[INDEXTTS CMD]", shlex.join(args), flush=True)
    return args

def synthesize_to_wav_bytes(
    text: str,
    ref_audio_path: str | None = None,
    emotion: str | None = None,              # kept for API compatibility (unused by CLI)
    target_duration_s: float | None = None,  # kept for API compatibility (unused by CLI)
) -> bytes:
    load()
    with tempfile.TemporaryDirectory() as td:
        out_wav = str(Path(td) / "out.wav")
        cmd = _build_cli(text, out_wav, ref_audio_path)
        # If the CLI supports emotion/duration in the future, add flags here.
        subprocess.run(cmd, check=True)
        return Path(out_wav).read_bytes()