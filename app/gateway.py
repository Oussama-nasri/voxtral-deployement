"""
Voxtral Gateway API
===================
REST + WebSocket proxy for vLLM's /v1/realtime Voxtral endpoint.

Correct vLLM Voxtral realtime protocol:
  1. Client connects to ws://.../v1/realtime
  2. Server sends {"type": "session.created", "id": "sess-..."}  ← MUST wait for this
  3. Client sends {"type": "session.update", "model": "<model_id>"}
  4. Client sends {"type": "input_audio_buffer.commit"}           ← signal ready
  5. Client sends {"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}  (repeat)
  6. Client sends {"type": "input_audio_buffer.commit", "final": true}  ← end of audio
  7. Server streams {"type": "transcription.delta", "delta": "..."}
  8. Server sends   {"type": "transcription.done",  "text": "...", "usage": {...}}
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import Optional

import aiohttp
import numpy as np
import soundfile as sf
import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

VLLM_HOST  = os.getenv("VLLM_HOST", "voxtral-server")
VLLM_PORT  = int(os.getenv("VLLM_PORT", "8000"))
MODEL_ID   = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
SAMPLE_RATE = 16_000
CHUNK_BYTES = 4096   # ~128ms per chunk at 16kHz PCM16 (matches official example)

app = FastAPI(
    title="Voxtral Mini 4B Realtime Gateway",
    description="REST + WebSocket gateway for Voxtral real-time speech transcription",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TranscriptionResponse(BaseModel):
    transcript: str
    duration_seconds: float
    processing_time_seconds: float
    language_detected: Optional[str] = None


def load_and_resample(audio_bytes: bytes) -> np.ndarray:
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


async def transcribe_via_vllm(audio: np.ndarray) -> str:
    """
    Stream PCM16 audio to vLLM /v1/realtime and return the full transcript.
    Follows the exact protocol from the official Voxtral vLLM example.
    """
    uri = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"

    async with websockets.connect(uri, ping_interval=20, max_size=10*1024*1024) as ws:

        # ── Step 1: Wait for session.created ──────────────────────────────────
        raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        event = json.loads(raw)
        if event.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got: {event}")
        session_id = event.get("id", "?")
        logger.info(f"Session created: {session_id}")

        # ── Step 2: Send session.update with model ────────────────────────────
        await ws.send(json.dumps({"type": "session.update", "model": MODEL_ID}))

        # ── Step 3: Initial commit (signal ready) ─────────────────────────────
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # ── Step 4: Stream audio in 4KB chunks ───────────────────────────────
        audio_bytes = audio.tobytes()
        total_chunks = (len(audio_bytes) + CHUNK_BYTES - 1) // CHUNK_BYTES
        logger.info(f"Sending {total_chunks} audio chunks ({len(audio_bytes)} bytes)")

        for i in range(0, len(audio_bytes), CHUNK_BYTES):
            chunk = audio_bytes[i: i + CHUNK_BYTES]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }))

        # ── Step 5: Final commit — signals end of audio ───────────────────────
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        logger.info("All audio sent. Waiting for transcription...")

        # ── Step 6: Collect transcript events ────────────────────────────────
        transcript_parts = []
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type", "")

            if etype == "transcription.delta":
                delta = event.get("delta", "")
                transcript_parts.append(delta)
                logger.debug(f"Delta: {delta!r}")

            elif etype == "transcription.done":
                # Use the full text from done event (most reliable)
                full = event.get("text", "".join(transcript_parts))
                logger.info(f"Transcription done. Tokens: {event.get('usage', {})}")
                return full

            elif etype == "error":
                raise RuntimeError(f"vLLM error: {event.get('error', event)}")

            else:
                logger.debug(f"Ignored event: {etype}")

    return "".join(transcript_parts)


# ── Routes ────────────────────────────────────────────────────────────────────

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
    """Upload an audio file (WAV/MP3/FLAC/OGG/M4A) → transcript."""
    t0 = time.time()
    raw = await file.read()

    try:
        audio = load_and_resample(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")

    duration = len(audio) / SAMPLE_RATE

    try:
        transcript = await transcribe_via_vllm(audio)
    except Exception as e:
        logger.error(f"vLLM error: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    proc_time = time.time() - t0
    logger.info(f"Transcribed {duration:.1f}s audio in {proc_time:.2f}s (RTF {proc_time/duration:.2f}x)")

    return TranscriptionResponse(
        transcript=transcript,
        duration_seconds=round(duration, 2),
        processing_time_seconds=round(proc_time, 2),
    )


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    Live microphone streaming proxy.
    Client sends raw PCM16 binary frames.
    Server relays transcript events as JSON:
      {"type": "delta",     "text": "..."}
      {"type": "completed", "text": "..."}
      {"type": "error",     "detail": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    vllm_uri = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"

    try:
        async with websockets.connect(vllm_uri, ping_interval=20) as vllm_ws:

            # Wait for session.created
            raw = await asyncio.wait_for(vllm_ws.recv(), timeout=15.0)
            event = json.loads(raw)
            if event.get("type") != "session.created":
                await websocket.send_json({"type": "error", "detail": f"Bad handshake: {event}"})
                return
            logger.info(f"vLLM session: {event.get('id')}")

            # session.update + initial commit
            await vllm_ws.send(json.dumps({"type": "session.update", "model": MODEL_ID}))
            await vllm_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            async def relay_from_vllm():
                async for raw in vllm_ws:
                    event = json.loads(raw)
                    etype = event.get("type", "")
                    if etype == "transcription.delta":
                        await websocket.send_json({"type": "delta", "text": event.get("delta", "")})
                    elif etype == "transcription.done":
                        await websocket.send_json({"type": "completed", "text": event.get("text", "")})
                    elif etype == "error":
                        await websocket.send_json({"type": "error", "detail": str(event.get("error"))})

            relay_task = asyncio.create_task(relay_from_vllm())

            try:
                while True:
                    data = await websocket.receive_bytes()
                    b64 = base64.b64encode(data).decode()
                    await vllm_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": b64,
                    }))
            except WebSocketDisconnect:
                logger.info("Client disconnected — sending final commit")
                await vllm_ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
                await asyncio.sleep(3)
            finally:
                relay_task.cancel()

    except Exception as e:
        logger.error(f"WS proxy error: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body style="font-family:monospace;padding:2rem;background:#0d1117;color:#e6edf3">
    <h2>🎙 Voxtral Mini 4B Realtime Gateway v2</h2>
    <ul>
      <li><code>GET  /health</code>     — health check</li>
      <li><code>GET  /models</code>     — list vLLM models</li>
      <li><code>POST /transcribe</code> — upload audio file</li>
      <li><code>WS   /ws/stream</code>  — live mic stream</li>
      <li><a href="/docs" style="color:#58a6ff">/docs</a> — Swagger UI</li>
    </ul></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")