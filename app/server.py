import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from app.model_runner import load, synthesize_to_wav_bytes

app = FastAPI(title="IndexTTS2 API (Local)", version="0.1.0")

@app.on_event("startup")
async def _startup():
    load()

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/tts")
async def tts(
    text: str = Form(...),
    emotion: str | None = Form(None),
    target_duration_s: float | None = Form(None),
    ref_audio: UploadFile | None = None
):
    ref_path = None
    if ref_audio:
        ref_path = f"/tmp/{ref_audio.filename}"
        with open(ref_path, "wb") as f:
            f.write(await ref_audio.read())

    wav_bytes = synthesize_to_wav_bytes(text, ref_path, emotion, target_duration_s)
    return JSONResponse({
        "format": "wav",
        "sr": 22050,
        "audio_base64": base64.b64encode(wav_bytes).decode("utf-8")
    })

# run locally with:
#   uvicorn app.server:app --host 0.0.0.0 --port 8000