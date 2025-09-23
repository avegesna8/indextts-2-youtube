from runpod import serverless
import base64
from app.model_runner import load, synthesize_to_wav_bytes

load()

def handler(event: dict):
    body = event.get("input", {}) or {}
    text = body.get("text", "")
    emotion = body.get("emotion")
    duration = body.get("duration")
    ref_audio_b64 = body.get("ref_audio_b64")

    ref_path = None
    if ref_audio_b64:
        import tempfile, base64 as b64
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(b64.b64decode(ref_audio_b64))
        tmp.flush()
        ref_path = tmp.name

    wav_bytes = synthesize_to_wav_bytes(
        text=text, ref_audio_path=ref_path, emotion=emotion, target_duration_s=duration
    )
    return {
        "format": "wav",
        "sr": 22050,
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8"),
    }

serverless.start({"handler": handler})
