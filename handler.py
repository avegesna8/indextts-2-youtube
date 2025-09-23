# handler.py
from runpod import serverless
import os, base64, tempfile, time, traceback

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "tts").lower()

_runner = None
def _lazy_runner():
    global _runner
    if _runner is None:
        from app.model_runner import load, synthesize_to_wav_bytes
        load()  # sanity check weights/config exist
        _runner = synthesize_to_wav_bytes
    return _runner

def _ok(data, extra=None):
    out = {"ok": True, **data}
    if extra: out.update(extra)
    return out

def _err(msg, code="INTERNAL", detail=None):
    return {"ok": False, "error": {"code": code, "message": msg, "detail": detail}}

def handler(event: dict):
    t0 = time.time()
    body = (event or {}).get("input", {}) or {}
    mode = (body.get("mode") or DEFAULT_MODE).lower()

    text = body.get("text", "")
    if not isinstance(text, str):
        return _err("`text` must be a string.", "BAD_REQUEST")

    # ---- Echo path (fast, no weights needed) ----
    if mode == "echo":
        return _ok({"echo": text, "message": f"You said: {text}"},
                   {"mode": "echo", "t_ms": int((time.time()-t0)*1000)})

    # ---- TTS path ----
    try:
        synth = _lazy_runner()

        ref_b64  = body.get("ref_audio_b64")
        ref_path = None
        if ref_b64:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(base64.b64decode(ref_b64))
            tmp.flush()
            ref_path = tmp.name

        wav_bytes = synth(
            text=text,
            ref_audio_path=ref_path,
            emotion=body.get("emotion"),
            target_duration_s=body.get("duration") or body.get("target_duration_s")
        )

        return _ok({
            "format": "wav",
            "sr": 22050,
            "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")
        }, {"mode": "tts", "t_ms": int((time.time()-t0)*1000)})

    except Exception as e:
        tb = traceback.format_exc()
        print("[TTS ERROR]", e, "\n", tb, flush=True)
        return _err("Synthesis failed. Check Logs.", "SYNTH_ERROR", str(e))

# Start RunPod serverless loop
serverless.start({"handler": handler})