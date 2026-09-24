# IndexTTS-2.5 RunPod worker

This is a separate upgrade of the old worker. The upstream code and model
revisions are pinned in `upstream.json` and the Dockerfile. It uses upstream's
locked Python dependencies, CUDA-enabled Torch, and the 2.5 model weights.

Build on Linux amd64 (or a remote builder):

```sh
docker build --platform linux/amd64 -t indextts25-test .
```

For the existing GitHub-connected RunPod deployment, place this Dockerfile and
handler.py at the repository root. Test the upgrade branch in a separate RunPod
endpoint before promoting it. The image downloads model weights at build time;
additional upstream auxiliary weights may download during first inference.
Use a CUDA 12.8-compatible driver and a BF16-capable GPU. No GPU inference has
been validated locally. CPU request-contract tests are in tests/.

Input retains `mode: tts`, `text`, and `ref_audio_b64`. New optional fields:
- `lang`: EN (default), ZH, JA, ES, AR.
- `duration_factor`: 0.5–2.0, default 1.0; 1.11 is slightly slower.
- `emotion`: optional text description, separate from spoken text.
- `emotion_strength`: 0–1, default 0.5.

Text emotion guidance explicitly enables upstream's Qwen emotion model.
These controls apply to the whole request. Per-phrase emotion planning and
stitching are not implemented by this upgrade.

A `mode: health` request reports version/capabilities without loading the GPU
model; it is not a synthesis test. A real short speech request must also pass.
Output includes model_version=2.5, actual WAV sample rate, and base64 audio.

Only after deploying and testing the 2.5 endpoint, update the video project's
.env:

```dotenv
RUNPOD_ENDPOINT_ID=THE_TESTED_ENDPOINT_ID
RUNPOD_MODEL_VERSION=2.5
RUNPOD_DURATION_FACTOR=1.11
RUNPOD_EMOTION=Curious and conversational, with subtle excitement
RUNPOD_EMOTION_STRENGTH=0.5
```

The client fingerprints these settings so older audio is not silently reused.
Leave RUNPOD_MODEL_VERSION unset/2 for the original endpoint. Roll back by
restoring its endpoint ID and version 2. No live endpoint has been changed by
preparing these files.
