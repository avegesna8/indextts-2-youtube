# app/model_runner.py
import os, shlex, subprocess, tempfile, sys, json
from pathlib import Path
from typing import Optional

# ---- Config/paths ----
REPO_ROOT = Path("/app/index-tts")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/index-tts/checkpoints"))
INFER_MODULE = os.getenv("INFER_MODULE", "indextts.infer_v2")  # v2 entry

def _log(msg: str, **kv):
    """Uniform, flushy logger."""
    blob = {"msg": msg, **kv}
    print("[model_runner]", json.dumps(blob, ensure_ascii=False), flush=True)

def _require(path: Path, desc: str):
    if not path.exists():
        _log("MISSING_PATH", desc=desc, path=str(path))
        raise FileNotFoundError(f"Missing {desc}: {path}")
    _log("FOUND_PATH", desc=desc, path=str(path))

def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

def _ffmpeg_info():
    exe = _which("ffmpeg")
    if not exe:
        _log("FFMPEG_NOT_IN_PATH")
        return
    try:
        out = subprocess.run([exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        first = (out.stdout or "").splitlines()[:2]
        _log("FFMPEG_OK", exe=exe, head=first)
    except Exception as e:
        _log("FFMPEG_CHECK_FAILED", error=repr(e))

def _ls(path: Path, limit: int = 50):
    try:
        names = sorted(os.listdir(path))[:limit]
        _log("LS", dir=str(path), count=len(names), sample=names)
    except Exception as e:
        _log("LS_FAILED", dir=str(path), error=repr(e))

def load() -> None:
    """Validate repo + checkpoints exist (and print lots of context)."""
    _log("ENV", PYTHON=sys.version, CWD=os.getcwd(), REPO_ROOT=str(REPO_ROOT), MODEL_DIR=str(MODEL_DIR),
         INFER_MODULE=INFER_MODULE)
    _ffmpeg_info()
    _require(REPO_ROOT / "indextts" / "infer_v2.py", "indextts/infer_v2.py")
    _require(MODEL_DIR, "MODEL_DIR (checkpoints directory)")
    _ls(MODEL_DIR)
    _require(MODEL_DIR / "config.yaml", "config.yaml in checkpoints")
    for fname in ["bpe.model", "gpt.pth", "s2mel.pth", "feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt"]:
        p = MODEL_DIR / fname
        _log("CHECK_FILE", file=fname, exists=p.exists(), size=(p.stat().st_size if p.exists() else None))

def _default_ref() -> Optional[str]:
    for p in [
        REPO_ROOT / "examples" / "voice_01.wav",
        REPO_ROOT / "examples" / "voice_02.wav",
        REPO_ROOT / "examples" / "voice_03.wav",
        REPO_ROOT / "test_data" / "input.wav",
    ]:
        if p.exists():
            _log("DEFAULT_REF_FOUND", path=str(p))
            return str(p)
    _log("DEFAULT_REF_NOT_FOUND")
    return None

def _probe_audio(path: str):
    """Log what this file looks like using soundfile and ffprobe."""
    info = {}
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            info.update(sr=f.samplerate, ch=f.channels, frames=len(f), fmt=f.format, subtype=f.subtype)
            _log("SF_INFO", path=path, **info)
    except Exception as e:
        _log("SF_INFO_FAILED", path=path, error=repr(e))
    # ffprobe best-effort
    exe = _which("ffprobe")
    if exe:
        try:
            proc = subprocess.run(
                [exe, "-v", "error", "-show_entries", "format=filename,format_name,duration", "-of", "json", path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            _log("FFPROBE", path=path, out=proc.stdout.strip()[:1000])
        except Exception as e:
            _log("FFPROBE_FAILED", path=path, error=repr(e))

def _to_clean_wav(src_path: str) -> str:
    """
    Transcode any input to a safe PCM16 mono 22050 Hz WAV so librosa/libsndfile/audioread are happy.
    """
    td = tempfile.mkdtemp()
    dst = str(Path(td) / "ref_clean.wav")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", src_path, "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", dst]
    _log("FFMPEG_TRANSCODE", cmd=" ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        _log("FFMPEG_TRANSCODE_FAILED", rc=res.returncode, stderr=res.stderr[-1000:])
        raise RuntimeError(f"ffmpeg failed: {res.stderr}")
    _log("FFMPEG_TRANSCODE_OK", dst=dst, size=os.path.getsize(dst))
    _probe_audio(dst)
    return dst

def _build_cli(text: str, out_wav: str, ref_audio: Optional[str]) -> list[str]:
    args = [
        "python3", "-m", INFER_MODULE,
        text,
        "--model_dir", str(MODEL_DIR),
        "--config", str(MODEL_DIR / "config.yaml"),
        "--output", out_wav,
    ]
    # choose a ref (payload -> default sample), then normalize it
    ref = ref_audio or _default_ref()
    if ref:
        _log("REF_INPUT", ref=ref)
        try:
            _probe_audio(ref)  # log what we got
            clean = _to_clean_wav(ref)
            args += ["--ref", clean]
        except Exception as e:
            _log("REF_NORMALIZE_FAILED", error=repr(e))
    else:
        _log("NO_REF", note="continuing without reference; model may error")

    _log("CLI", cmd=shlex.join(args), cwd=str(REPO_ROOT))
    return args

def synthesize_to_wav_bytes(
    text: str,
    ref_audio_path: Optional[str] = None,
    emotion: Optional[str] = None,             # unused by CLI; logged for visibility
    target_duration_s: Optional[float] = None, # unused by CLI; logged for visibility
) -> bytes:
    load()
    _log("SYNTH_START", text_len=len(text), has_ref=bool(ref_audio_path),
         emotion=emotion, target_duration_s=target_duration_s)
    with tempfile.TemporaryDirectory() as td:
        out_wav = str(Path(td) / "out.wav")
        cmd = _build_cli(text, out_wav, ref_audio_path)
        # capture stdout/stderr so we can print on failure
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _log("CLI_FINISHED", rc=proc.returncode)
        if proc.stdout:
            _log("CLI_STDOUT", tail=proc.stdout[-1000:])
        if proc.stderr:
            _log("CLI_STDERR", tail=proc.stderr[-2000:])
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)

        size = os.path.getsize(out_wav) if os.path.exists(out_wav) else -1
        _log("SYNTH_DONE", out=out_wav, size=size)
        return Path(out_wav).read_bytes()