"""
Voxtral Gateway API
===================
A thin FastAPI service that sits in front of vLLM's realtime WebSocket endpoint
and exposes a simpler REST interface:

  POST /transcribe      - Upload an audio file, get back a transcript
  GET  /health          - Service health + vLLM status
  GET  /models          - List available models from vLLM
  WS   /ws/stream       - Raw WebSocket proxy for realtime microphone streaming
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import aiohttp
import aiofiles
import numpy as np
import soundfile as sf
import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── Config ───────────────────────────────────────────────────────────────────
VLLM_HOST  = os.getenv("VLLM_HOST", "voxtral-server")
VLLM_PORT  = int(os.getenv("VLLM_PORT", "8000"))
MODEL_ID   = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.1)   # 100ms chunks

app = FastAPI(
    title="Voxtral Mini 4B Realtime Gateway",
    description="REST + WebSocket gateway for Voxtral real-time speech transcription",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────

class TranscriptionResponse(BaseModel):
    transcript: str
    duration_seconds: float
    processing_time_seconds: float
    language_detected: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_and_resample(audio_bytes: bytes, orig_format: str = None) -> np.ndarray:
    """Load audio bytes, convert to int16 mono @ 16kHz."""
    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype="int16")
    except Exception:
        import librosa
        buf.seek(0)
        audio_f, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
        return (audio_f * 32767).astype(np.int16)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != SAMPLE_RATE:
        import librosa
        audio_f = audio.astype(np.float32) / 32768.0
        audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = (audio_f * 32767).astype(np.int16)

    return audio


async def stream_audio_to_vllm(audio: np.ndarray) -> str:
    """Connect to vLLM realtime WebSocket, stream audio, collect transcript."""
    uri = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime?model={MODEL_ID}"
    transcript_parts: list[str] = []
    completed = asyncio.Event()

    async with websockets.connect(uri, ping_interval=20, max_size=10*1024*1024) as ws:
        # Initialize session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "model": MODEL_ID,
                "temperature": 0.0,
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": MODEL_ID,
                    "language": "auto",
                },
            }
        }))

        async def receiver():
            async for raw in ws:
                event = json.loads(raw)
                etype = event.get("type", "")
                if etype == "conversation.item.input_audio_transcription.delta":
                    transcript_parts.append(event.get("delta", ""))
                elif etype == "conversation.item.input_audio_transcription.completed":
                    completed.set()
                elif etype == "error":
                    logger.error(f"vLLM error: {event}")
                    completed.set()

        recv_task = asyncio.create_task(receiver())

        # Wait for session confirmation
        await asyncio.sleep(0.3)

        # Stream PCM chunks
        total = len(audio)
        for i in range(0, total, CHUNK_SAMPLES):
            chunk = audio[i: i + CHUNK_SAMPLES]
            b64 = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))
            await asyncio.sleep(0.01)   # mild backpressure

        # Signal end of audio
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait for transcript completion (max 60s)
        try:
            await asyncio.wait_for(completed.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Transcript completion timeout – returning partial result")

        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    return "".join(transcript_parts)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    vllm_ok = False
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(
                f"http://{VLLM_HOST}:{VLLM_PORT}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                vllm_ok = resp.status == 200
    except Exception:
        pass

    return {
        "gateway": "ok",
        "vllm_backend": "ok" if vllm_ok else "unavailable",
        "model": MODEL_ID,
        "timestamp": time.time(),
    }


@app.get("/models")
async def list_models():
    async with aiohttp.ClientSession() as client:
        async with client.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models") as resp:
            return await resp.json()


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and receive a full transcript.
    Supported formats: WAV, MP3, FLAC, OGG, M4A, WebM.
    """
    t0 = time.time()
    raw = await file.read()

    try:
        audio = load_and_resample(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

    duration = len(audio) / SAMPLE_RATE

    try:
        transcript = await stream_audio_to_vllm(audio)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vLLM error: {e}")

    proc_time = time.time() - t0
    logger.info(f"Transcribed {duration:.1f}s audio in {proc_time:.2f}s")

    return TranscriptionResponse(
        transcript=transcript,
        duration_seconds=round(duration, 2),
        processing_time_seconds=round(proc_time, 2),
    )


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    WebSocket proxy for real-time microphone streaming.
    Client sends raw PCM16 frames; server proxies to vLLM and
    forwards transcript deltas back to the client as JSON.

    Client message: binary PCM16 audio bytes
    Server messages:
      { "type": "delta",     "text": "..." }
      { "type": "completed", "text": "..." }
      { "type": "error",     "detail": "..." }
    """
    await websocket.accept()
    logger.info("WebSocket client connected for live streaming")

    vllm_uri = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime?model={MODEL_ID}"

    try:
        async with websockets.connect(vllm_uri, ping_interval=20) as vllm_ws:
            # Initialize vLLM session
            await vllm_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "model": MODEL_ID,
                    "temperature": 0.0,
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": MODEL_ID,
                        "language": "auto",
                    },
                }
            }))

            async def forward_from_vllm():
                """Relay transcript events from vLLM to the browser client."""
                async for raw in vllm_ws:
                    event = json.loads(raw)
                    etype = event.get("type", "")
                    if etype == "conversation.item.input_audio_transcription.delta":
                        await websocket.send_json({
                            "type": "delta",
                            "text": event.get("delta", ""),
                        })
                    elif etype == "conversation.item.input_audio_transcription.completed":
                        await websocket.send_json({
                            "type": "completed",
                            "text": event.get("transcript", ""),
                        })

            relay_task = asyncio.create_task(forward_from_vllm())

            # Forward client audio to vLLM
            try:
                while True:
                    data = await websocket.receive_bytes()
                    b64 = base64.b64encode(data).decode()
                    await vllm_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64,
                    }))
            except WebSocketDisconnect:
                logger.info("Client disconnected – committing final audio buffer")
                await vllm_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                await asyncio.sleep(2)
            finally:
                relay_task.cancel()

    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#e6edf3">
    <h2>🎙 Voxtral Mini 4B Realtime Gateway</h2>
    <p>Endpoints:</p>
    <ul>
      <li><code>GET  /health</code>        — Service + backend health</li>
      <li><code>GET  /models</code>         — Available vLLM models</li>
      <li><code>POST /transcribe</code>     — Upload audio file → transcript</li>
      <li><code>WS   /ws/stream</code>      — Live microphone stream → transcript deltas</li>
      <li><a href="/docs" style="color:#58a6ff">/docs</a> — Swagger UI</li>
    </ul>
    </body></html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
