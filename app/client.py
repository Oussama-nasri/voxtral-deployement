"""
Voxtral Mini 4B Realtime - WebSocket Streaming Transcription Client
Connects to a vLLM /v1/realtime endpoint and streams audio for live transcription.
"""

import asyncio
import json
import os
import sys
import time
import wave
import struct
import logging
from pathlib import Path
from typing import Optional

import websockets
import soundfile as sf
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
VLLM_HOST       = os.getenv("VLLM_HOST", "voxtral-server")
VLLM_PORT       = int(os.getenv("VLLM_PORT", "8000"))
MODEL_ID        = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
SAMPLE_RATE     = 16_000          # Voxtral requires 16kHz
CHUNK_DURATION  = 0.1             # seconds per audio chunk sent to server
CHUNK_SAMPLES   = int(SAMPLE_RATE * CHUNK_DURATION)
TEMPERATURE     = 0.0             # always 0 per official recommendation


# ─── WebSocket Realtime Session ───────────────────────────────────────────────

class VoxtralRealtimeSession:
    """
    Manages a WebSocket session with vLLM's /v1/realtime endpoint.
    Sends PCM16 audio chunks and receives incremental transcript tokens.
    """

    def __init__(self, host: str = VLLM_HOST, port: int = VLLM_PORT):
        self.uri = f"ws://{host}:{port}/v1/realtime?model={MODEL_ID}"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self.full_transcript: list[str] = []
        self._recv_task: Optional[asyncio.Task] = None

    async def connect(self):
        logger.info(f"Connecting to vLLM Realtime endpoint: {self.uri}")
        self.ws = await websockets.connect(
            self.uri,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,  # 10MB max message
        )
        logger.info("WebSocket connection established.")
        await self._initialize_session()
        # Start background receiver
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def _initialize_session(self):
        """Send session.update to configure the realtime session."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "model": MODEL_ID,
                "temperature": TEMPERATURE,
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": MODEL_ID,
                    "language": "auto",         # auto-detect language
                },
            }
        }
        await self.ws.send(json.dumps(session_config))
        logger.info("Session configuration sent.")

    async def _receive_loop(self):
        """Background task: continuously receive and process server events."""
        try:
            async for raw_msg in self.ws:
                await self._handle_event(json.loads(raw_msg))
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket closed: {e.code} {e.reason}")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    async def _handle_event(self, event: dict):
        event_type = event.get("type", "")

        if event_type == "session.created":
            self.session_id = event.get("session", {}).get("id")
            logger.info(f"Session created: {self.session_id}")

        elif event_type == "session.updated":
            logger.debug("Session updated successfully.")

        elif event_type == "conversation.item.input_audio_transcription.delta":
            # Incremental transcript token
            delta = event.get("delta", "")
            if delta:
                print(delta, end="", flush=True)
                self.full_transcript.append(delta)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            logger.info(f"\n[Transcript complete] {transcript}")

        elif event_type == "input_audio_buffer.speech_started":
            logger.info("[VAD] Speech detected - transcription starting...")

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.info("[VAD] Speech ended.")

        elif event_type == "error":
            logger.error(f"Server error: {event.get('error', {})}")

        else:
            logger.debug(f"Unhandled event type: {event_type}")

    async def send_audio_chunk(self, pcm16_bytes: bytes):
        """
        Send a chunk of PCM16 audio (16kHz, mono, little-endian int16).
        The payload is base64-encoded per the OpenAI Realtime API spec.
        """
        import base64
        audio_b64 = base64.b64encode(pcm16_bytes).decode("utf-8")
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }
        await self.ws.send(json.dumps(message))

    async def commit_audio(self):
        """Signal end-of-utterance to trigger transcription."""
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()
        logger.info("Session closed.")

    def get_full_transcript(self) -> str:
        return "".join(self.full_transcript)


# ─── Audio File Transcription ─────────────────────────────────────────────────

async def transcribe_file(audio_path: str) -> str:
    """
    Transcribe a local audio file by streaming it to Voxtral via WebSocket.
    Supports WAV, MP3, FLAC, OGG, etc. (anything soundfile/librosa can read).
    """
    logger.info(f"Loading audio: {audio_path}")

    # Load and resample to 16kHz mono
    try:
        audio, sr = sf.read(audio_path, dtype="int16")
    except Exception:
        # Fallback to librosa for MP3/other formats
        import librosa
        audio_f, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        audio = (audio_f * 32767).astype(np.int16)
        sr = SAMPLE_RATE

    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if stereo

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        audio_f = audio.astype(np.float32) / 32768.0
        audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = (audio_f * 32767).astype(np.int16)

    total_samples = len(audio)
    total_duration = total_samples / SAMPLE_RATE
    logger.info(f"Audio: {total_duration:.1f}s, {total_samples} samples @ {SAMPLE_RATE}Hz")

    session = VoxtralRealtimeSession()
    try:
        await session.connect()
        # Wait for session initialization
        await asyncio.sleep(0.5)

        logger.info("Streaming audio chunks...")
        print("\n--- TRANSCRIPT ---\n", flush=True)

        # Stream audio in chunks
        for i in range(0, total_samples, CHUNK_SAMPLES):
            chunk = audio[i: i + CHUNK_SAMPLES]
            pcm_bytes = chunk.tobytes()
            await session.send_audio_chunk(pcm_bytes)
            # Simulate real-time pacing
            await asyncio.sleep(CHUNK_DURATION * 0.9)

        # Commit the full buffer
        await session.commit_audio()

        # Wait for remaining transcript tokens
        await asyncio.sleep(3.0)

        transcript = session.get_full_transcript()
        print(f"\n\n--- FINAL TRANSCRIPT ---\n{transcript}\n")
        return transcript

    finally:
        await session.close()


# ─── Microphone Streaming ──────────────────────────────────────────────────────

async def transcribe_microphone():
    """
    Live microphone transcription using PyAudio + WebSocket streaming.
    Runs until Ctrl+C.
    """
    try:
        import pyaudio
    except ImportError:
        logger.error("PyAudio not installed. Run: pip install pyaudio")
        return

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )

    session = VoxtralRealtimeSession()
    try:
        await session.connect()
        await asyncio.sleep(0.5)

        logger.info("Microphone streaming started. Press Ctrl+C to stop.")
        print("\n--- LIVE TRANSCRIPT ---\n", flush=True)

        while True:
            pcm_bytes = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            await session.send_audio_chunk(pcm_bytes)
            await asyncio.sleep(0)

    except KeyboardInterrupt:
        logger.info("Stopping microphone capture...")
        await session.commit_audio()
        await asyncio.sleep(2.0)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        await session.close()


# ─── Health Check ─────────────────────────────────────────────────────────────

async def wait_for_server(max_wait: int = 300):
    """Poll the vLLM health endpoint until ready."""
    import aiohttp
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
    logger.info(f"Waiting for vLLM server at {url} ...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        logger.info("vLLM server is ready!")
                        return True
        except Exception:
            pass
        await asyncio.sleep(5)
    raise TimeoutError(f"vLLM server not ready after {max_wait}s")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voxtral Mini 4B Realtime Client")
    parser.add_argument("--mode", choices=["file", "mic", "healthcheck"],
                        default="healthcheck", help="Operation mode")
    parser.add_argument("--audio", type=str, default=None,
                        help="Path to audio file (for --mode file)")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for vLLM server to be ready before starting")
    args = parser.parse_args()

    async def main():
        if args.wait:
            await wait_for_server()

        if args.mode == "healthcheck":
            await wait_for_server(max_wait=10)
        elif args.mode == "file":
            if not args.audio:
                logger.error("--audio PATH required for file mode")
                sys.exit(1)
            await transcribe_file(args.audio)
        elif args.mode == "mic":
            await transcribe_microphone()

    asyncio.run(main())
