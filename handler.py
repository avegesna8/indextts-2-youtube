"""RunPod worker for pinned IndexTTS-2.5; compatible with the existing WAV contract."""
import base64
import binascii
import io
import math
import os
from pathlib import Path
import tempfile
import threading
import traceback
import wave

MODEL_VERSION = "2.5"
_model = None
_lock = threading.Lock()


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return value


def options(body):
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string.")
    lang = body.get("lang", "EN")
    if lang not in ("EN", "ZH", "JA", "ES", "AR"):
        raise ValueError("Unsupported lang.")
    emotion = body.get("emotion")
    if emotion is not None and (not isinstance(emotion, str) or not emotion.strip() or len(emotion) > 1000):
        raise ValueError("emotion must be a nonempty description under 1000 characters.")
    return dict(text=text, lang=lang,
                duration_factor=number(body.get("duration_factor", 1.0), "duration_factor", 0.5, 2.0),
                use_emo_text=bool(emotion), emo_text=emotion,
                emo_alpha=number(body.get("emotion_strength", 0.5), "emotion_strength", 0, 1),
                use_random=False, verbose=False)


def model_directory():
    root = Path(os.getenv("INDEXTTS25_MODEL_DIR") or os.getenv("MODEL_DIR", "/models/indextts25"))
    # Existing RunPod endpoint variables survive image upgrades.
    if str(root) in ("/app/index-tts/checkpoints", "/models/indextts2"):
        print(f"Ignoring legacy IndexTTS-2 MODEL_DIR={root}; using bundled 2.5 weights.", flush=True)
        root = Path("/models/indextts25")
    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(f"Missing {root / 'config.yaml'}. Set INDEXTTS25_MODEL_DIR=/models/indextts25; check endpoint environment and volume mounts.")
    return str(root)


def get_model():
    global _model
    if _model is None:
        from indextts.infer_v2_5 import IndexTTS2
        root = model_directory()
        _model = IndexTTS2(cfg_path=f"{root}/config.yaml", model_dir=root,
                          use_bf16=True, use_cuda_kernel=False, use_qwen_emo=True)
    return _model


def handler(event):
    try:
        body = event.get("input", {})
        if not isinstance(body, dict):
            raise ValueError("input must be an object.")
        if body.get("mode") == "health":
            return {"ok": True, "model_version": MODEL_VERSION,
                    "capabilities": ["emotion", "duration_factor"], "model_loaded": _model is not None}
        kwargs = options(body)
        encoded = body.get("ref_audio_b64")
        if not isinstance(encoded, str) or len(encoded) > 7 * 1024 * 1024:
            raise ValueError("A reference audio clip under 5 MB is required.")
        reference = base64.b64decode(encoded, validate=True)
        if not reference or len(reference) > 5 * 1024 * 1024:
            raise ValueError("A reference audio clip under 5 MB is required.")
        with _lock, tempfile.TemporaryDirectory() as tmp:
            ref, out = Path(tmp) / "reference.wav", Path(tmp) / "output.wav"
            ref.write_bytes(reference)
            get_model().infer(spk_audio_prompt=str(ref), output_path=str(out), **kwargs)
            data = out.read_bytes()
            with wave.open(io.BytesIO(data), "rb") as wav:
                if wav.getnframes() == 0:
                    raise RuntimeError("Model produced empty audio.")
                sr = wav.getframerate()
        return {"ok": True, "format": "wav", "sr": sr, "model_version": MODEL_VERSION,
                "audio_base64": base64.b64encode(data).decode()}
    except (ValueError, binascii.Error) as exc:
        return {"ok": False, "error": {"code": "BAD_REQUEST", "message": str(exc)}}
    except Exception as exc:
        print(f"IndexTTS-2.5 synthesis failed: {type(exc).__name__}", flush=True)
        traceback.print_exc()
        return {"ok": False, "error": {"code": "SYNTH_ERROR", "message": "Synthesis failed; inspect worker logs."}}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
