import asyncio
import io
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# Global lock to serialize TTS operations as pyttsx3 is not thread-safe
_tts_lock = asyncio.Lock()

def _synthesize_speech_sync(text: str) -> bytes:
    """
    Synchronous implementation of TTS using pyttsx3.
    Must be run in a thread.
    """
    import pyttsx3
    try:
        engine = pyttsx3.init()
    except Exception as e:
        logger.error(f"Failed to initialize pyttsx3: {e}")
        raise

    # Configure voice/rate optionally
    # Default to 175 as used in the codebase
    engine.setProperty('rate', 175)

    # pyttsx3 doesn't write to BytesIO directly; we can write to a temp file then read
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        temp_path = tf.name

    try:
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        with open(temp_path, 'rb') as f:
            data = f.read()
        return data
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

async def synthesize_speech(text: str) -> bytes:
    """
    Synthesize speech locally using pyttsx3 (offline).
    Offloads to a thread and uses a lock to ensure thread safety.
    Returns WAV bytes.
    """
    async with _tts_lock:
        return await asyncio.to_thread(_synthesize_speech_sync, text)
