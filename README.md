# Voxtral Mini 4B Realtime — Dockerized Deployment

## What is Voxtral Mini 4B Realtime?

Mistral AI's **Voxtral Mini 4B Realtime 2602** is a **multilingual, real-time speech transcription model** — one of the first open-source models to match offline accuracy at sub-500ms latency.

| Attribute        | Detail                                       |
|------------------|----------------------------------------------|
| Architecture     | ~3.4B LM + ~970M causal audio encoder        |
| Languages        | 13 (Arabic, French, English, German, Spanish, Hindi, Italian, Dutch, Portuguese, Chinese, Japanese, Korean, Russian) |
| Delay options    | 80ms → 2400ms (configurable, default **480ms**) |
| Format           | BF16, Apache-2.0 license                     |
| Recommended GPU  | ≥ 16 GB VRAM (A4000, 3090, A10, etc.)        |
| Inference engine | **vLLM** (production-recommended)            |

---

## Stack Architecture

```
Browser / CLI
     │
     ▼
┌──────────────────────┐   REST / WebSocket
│  voxtral-gateway     │◄──────────────────── your app
│  FastAPI :7860       │
│  (CPU-only)          │
└──────────┬───────────┘
           │  WebSocket (ws://voxtral-server:8000/v1/realtime)
           ▼
┌──────────────────────┐
│  voxtral-server      │
│  vLLM :8000          │
│  RTX A4000 (GPU)     │
└──────────────────────┘
```

---

## Prerequisites

On your A4000 server:

```bash
# 1. Docker Engine 24+
curl -fsSL https://get.docker.com | bash

# 2. NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Quick Start

```bash
# 1. Clone / copy this project
cd voxtral-docker

# 2. Set your HuggingFace token
cp .env.example .env
# Edit .env: HF_TOKEN=hf_your_token_here
# (Get token at https://huggingface.co/settings/tokens)
# (Accept model terms at https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)

# 3. Start everything
bash scripts/run.sh up
#  → Builds images, starts services, waits until ready
#  → First run downloads ~8 GB of model weights (cached in Docker volume)

# 4. Transcribe a file
bash scripts/run.sh transcribe /path/to/audio.wav

# 5. Check health
bash scripts/run.sh health

# 6. Check GPU
bash scripts/run.sh gpu
```

---

## API Usage

### REST — Transcribe a file

```bash
curl -X POST http://localhost:7860/transcribe \
  -F "file=@meeting.wav" \
  | jq .
```

Response:
```json
{
  "transcript": "Hello, this is a test transcription.",
  "duration_seconds": 3.5,
  "processing_time_seconds": 1.2
}
```

### Python — Transcribe a file

```python
import requests

with open("audio.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:7860/transcribe",
        files={"file": ("audio.wav", f, "audio/wav")}
    )
print(resp.json()["transcript"])
```

### Python — Live WebSocket Streaming

```python
import asyncio, websockets, json, base64
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16_000
CHUNK = 1600  # 100ms

async def stream_mic():
    uri = "ws://localhost:7860/ws/stream"
    async with websockets.connect(uri) as ws:
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()

        def callback(indata, frames, time, status):
            loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1,
                               dtype="int16", blocksize=CHUNK,
                               callback=callback):
            print("Listening... (Ctrl+C to stop)")
            async def sender():
                while True:
                    data = await q.get()
                    await ws.send(data)

            async def receiver():
                async for msg in ws:
                    event = json.loads(msg)
                    if event["type"] == "delta":
                        print(event["text"], end="", flush=True)

            await asyncio.gather(sender(), receiver())

asyncio.run(stream_mic())
```

### Direct vLLM WebSocket (advanced)

The vLLM server exposes the OpenAI Realtime API at:
```
ws://localhost:8000/v1/realtime?model=mistralai/Voxtral-Mini-4B-Realtime-2602
```

See `app/client.py` for a complete reference implementation.

---

## Configuration

### Transcription Delay

Edit `tekken.json` inside the model weights (in the Docker volume) to change delay:

```json
{ "transcription_delay_ms": 480 }
```

Valid values: multiples of 80 between 80–1200, or 2400.
- **160ms** — lowest latency, slightly higher WER
- **480ms** — recommended sweet spot  
- **2400ms** — near-offline quality

### vLLM Flags (in `Dockerfile.server`)

| Flag | Default | Notes |
|------|---------|-------|
| `--max-model-len` | 45000 | ~1h audio. Set to 131072 for 3h (needs more VRAM) |
| `--gpu-memory-utilization` | 0.90 | Leave ~1.6 GB headroom on 16 GB A4000 |
| `--dtype` | bfloat16 | Matches model weights |
| `--max-num-batched-tokens` | (auto) | Increase for higher throughput, increases latency |

---

## Troubleshooting

**Model not found / 403 error:**
```bash
# Make sure you accepted terms at HuggingFace and token is correct
docker compose exec voxtral-server python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602')"
```

**Out of VRAM:**
```bash
# Reduce max-model-len in Dockerfile.server CMD
--max-model-len 20000  # ~27 min of audio
```

**CUDA compilation takes too long:**
Already handled via `VLLM_DISABLE_COMPILE_CACHE=1` + `PIECEWISE` cudagraph mode.

**Check GPU usage:**
```bash
bash scripts/run.sh gpu
# or
watch -n1 nvidia-smi
```

---

## Files

```
voxtral-docker/
├── app/
│   ├── client.py       # Standalone WebSocket client (file + mic modes)
│   └── gateway.py      # FastAPI REST + WebSocket proxy service
├── scripts/
│   └── run.sh          # Helper commands
├── Dockerfile.server   # vLLM inference server (GPU)
├── Dockerfile.gateway  # FastAPI gateway (CPU)
├── docker-compose.yml  # Full stack orchestration
├── requirements.txt    # Gateway Python deps
├── .env.example        # Environment variable template
└── README.md
```
