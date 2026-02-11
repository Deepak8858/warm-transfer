from __future__ import annotations

import asyncio
import os
from typing import Dict, Optional
import logging
import io
import tempfile
import wave
import numpy as np

logger = logging.getLogger(__name__)

# Feature flags
ENABLE_AGENT_MOCK = os.getenv("ENABLE_AGENT_MOCK", "1") == "1"
ENABLE_VOICE_AI = os.getenv("ENABLE_VOICE_AI", "0") == "1"

try:
    # LiveKit Agents / RTC imports for media publishing
    from livekit import rtc
    AGENTS_AVAILABLE = True
except Exception:
    rtc = None  # type: ignore
    AGENTS_AVAILABLE = False

# Try to import voice agent
try:
    from . import voice_agent
    VOICE_AI_AVAILABLE = True
except Exception:
    try:
        import voice_agent  # type: ignore
        VOICE_AI_AVAILABLE = True
    except Exception:
        voice_agent = None  # type: ignore
        VOICE_AI_AVAILABLE = False


class AgentSession:
    def __init__(self, room_name: str, identity: str = "ai-agent"):
        self.room_name = room_name
        self.identity = identity
        self.running = False
        self.last_prompt: Optional[str] = None
        # Placeholder for real agent resources
        self._task: Optional[asyncio.Task] = None
        self._room = None
        self._audio_track = None
        self._audio_source = None
        self._voice_agent_task = None

    async def start(self):
        self.running = True
        
        # Priority 1: Voice AI Agent (full conversational AI)
        if VOICE_AI_AVAILABLE and ENABLE_VOICE_AI and not ENABLE_AGENT_MOCK and voice_agent:
            logger.info(f"[Agent] Starting VOICE AI agent for room={self.room_name}")
            try:
                self._voice_agent_task = await voice_agent.start_agent_job(self.room_name, self.identity)
                return
            except Exception as e:
                logger.error(f"Failed to start voice AI agent: {e}")
                # Fall back to basic real agent
        
        # Priority 2: Basic Real Agent (TTS only)
        if AGENTS_AVAILABLE and not ENABLE_AGENT_MOCK:
            logger.info(f"[Agent] Starting REAL agent for room={self.room_name}")
            self._task = asyncio.create_task(self._run_real())
        else:
            # Priority 3: Mock Agent
            logger.info(f"[Agent] Starting MOCK agent for room={self.room_name}")
            self._task = asyncio.create_task(self._heartbeat())

    async def say(self, text: str):
        self.last_prompt = text
        logger.info(f"[Agent] say(room={self.room_name}): {text[:100]}")
        
        # Try voice AI agent first
        if VOICE_AI_AVAILABLE and ENABLE_VOICE_AI and self._voice_agent_task and voice_agent:
            try:
                await voice_agent.agent_say(self.room_name, text)
                return
            except Exception as e:
                logger.warning(f"Voice AI agent say failed: {e}")
        
        # Fall back to basic TTS
        if AGENTS_AVAILABLE and not ENABLE_AGENT_MOCK:
            await self._speak(text)

    async def stop(self):
        self.running = False
        
        # Stop voice AI agent if running
        if VOICE_AI_AVAILABLE and self._voice_agent_task and voice_agent:
            try:
                await voice_agent.stop_agent_job(self.room_name)
            except Exception as e:
                logger.warning(f"Error stopping voice AI agent: {e}")
        
        # Stop basic agent task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[Agent] Stopped for room={self.room_name}")

    async def _heartbeat(self):
        try:
            while self.running:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    async def _run_real(self):
        """Connect to LiveKit room and set up audio track for TTS playback."""
        # Import locally to avoid static analyzers complaining when rtc isn't available
        try:
            from livekit import rtc as _rtc
        except Exception:
            logger.error("livekit.rtc not available; cannot start real agent")
            self.running = False
            return
        from dotenv import load_dotenv
        load_dotenv()
        ws_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        if not (ws_url and api_key and api_secret):
            logger.error("Missing LiveKit env; cannot start real agent")
            self.running = False
            return

        # Build token using server SDK is on the main app; reuse HTTP GET /get-token if desired.
        # Here we rely on rtc.Room.connect with a token provided by the API.
        # To avoid cross-calls, we create a short-lived token inline using the same env.
        from livekit import api as lk_api
        token = (
            lk_api.AccessToken(api_key, api_secret)
            .with_identity(self.identity)
            .with_name(self.identity)
            .with_grants(lk_api.VideoGrants(room_join=True, room=self.room_name, can_publish=True, can_subscribe=True))
            .to_jwt()
        )

        room = _rtc.Room()
        self._room = room
        
        try:
            await room.connect(ws_url, token)
            logger.info(f"[Agent] Connected to room {self.room_name}")

            # Wait a moment for room connection to stabilize
            await asyncio.sleep(0.5)
            
            # Prepare an AudioSource at 48000 Hz mono as required by LiveKit
            self._audio_source = _rtc.AudioSource(sample_rate=48000, num_channels=1)
            self._audio_track = _rtc.LocalAudioTrack.create_audio_track("agent_tts", self._audio_source)
            if self._audio_track is not None:
                await room.local_participant.publish_track(self._audio_track)
                logger.info(f"[Agent] Published audio track for {self.room_name}")

            # Main loop
            while self.running:
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                if self._room:
                    await self._room.disconnect()
                    logger.info(f"[Agent] Disconnected from room {self.room_name}")
            except Exception as e:
                logger.warning(f"Error disconnecting from room: {e}")

    async def _speak(self, text: str):
        """Synthesize text to WAV and push PCM frames to audio source."""
        if not self._audio_source:
            logger.warning("Audio source not initialized; cannot speak")
            return
        try:
            from livekit import rtc as _rtc
        except Exception:
            logger.error("livekit.rtc not available; cannot speak")
            return
        try:
            try:
                from . import tts
            except ImportError:
                import tts

            # Use async TTS from tts.py
            wav_bytes = await tts.synthesize_speech(text)

            # Read WAV and stream to 48kHz mono float32/pcm16 frames
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                src_rate = wf.getframerate()
                src_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

                # Convert to numpy int16
                if sampwidth == 2:
                    audio = np.frombuffer(frames, dtype=np.int16)
                elif sampwidth == 1:
                    # 8-bit unsigned to int16
                    audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) << 8
                else:
                    # Fallback: assume 16-bit
                    audio = np.frombuffer(frames, dtype=np.int16)

                # If stereo, take mono by averaging channels
                if src_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

                # Resample to 48000 Hz if needed
                target_sr = 48000
                if src_rate != target_sr:
                    # Simple linear resample
                    duration = audio.shape[0] / src_rate
                    target_len = int(duration * target_sr)
                    x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
                    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
                    audio = np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.int16)

                # Chunk into ~20ms frames (960 samples at 48kHz)
                frame_size = 960
                idx = 0
                while idx < len(audio) and self.running:
                    chunk = audio[idx:idx+frame_size]
                    if len(chunk) < frame_size:
                        pad = np.zeros(frame_size - len(chunk), dtype=np.int16)
                        chunk = np.concatenate([chunk, pad])
                    # Push to LiveKit audio source
                    frame = _rtc.AudioFrame(
                        data=chunk.tobytes(),
                        sample_rate=target_sr,
                        num_channels=1,
                        samples_per_channel=len(chunk),
                    )
                    res = self._audio_source.capture_frame(frame)
                    if asyncio.iscoroutine(res):
                        await res
                    idx += frame_size
                    await asyncio.sleep(0.02)
        except Exception as e:
            logger.error(f"Agent speak error: {e}")


class AgentManager:
    def __init__(self):
        self.sessions: Dict[str, AgentSession] = {}

    async def start_agent(self, room_name: str, identity: str = "ai-agent"):
        if room_name in self.sessions and self.sessions[room_name].running:
            return
        sess = AgentSession(room_name, identity)
        self.sessions[room_name] = sess
        await sess.start()

    async def say(self, room_name: str, text: str):
        sess = self.sessions.get(room_name)
        if not sess or not sess.running:
            raise RuntimeError("Agent not running for this room")
        await sess.say(text)

    async def stop_agent(self, room_name: str):
        sess = self.sessions.get(room_name)
        if not sess or not sess.running:
            return
        await sess.stop()


manager = AgentManager()
