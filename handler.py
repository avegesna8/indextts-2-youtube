from runpod import serverless
import base64

from app.model_runner import load, synthesize_to_wav_bytes

# Load once per cold start
load()

def handler(event: dict):
    """
    RunPod passes {"input": {...}} here.
    Expected input:
      {
        "text": "hello world",
        "emotion": "neutral",         # optional
        "duration": 5.0,              # optional, seconds
        "ref_audio_b64": "<base64>"   # optional, for style/speaker (if you wire it)
      }
    """
    body = event.get("input", {})
    text = body.get("text", "")
    emotion = body.get("emotion")
    duration = body.get("duration")
    ref_audio_b64 = body.get("ref_audio_b64")

    ref_path = None
    if ref_audio_b64:
        import base64, tempfile
        ref_bytes = base64.b64decode(ref_audio_b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(ref_bytes)
        tmp.flush()
        ref_path = tmp.name

    wav_bytes = synthesize_to_wav_bytes(
        text=text,
        ref_audio_path=ref_path,
        emotion=emotion,
        target_duration_s=duration
    )
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return {"audio_base64": audio_b64, "format": "wav", "sr": 22050}

serverless.start({"handler": handler})
