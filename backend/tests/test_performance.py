"""
Performance test for blocking operations.
This test verifies that expensive operations like TTS do not block the main event loop.
It mocks pyttsx3 to simulate a blocking call and measures the loop responsiveness.
"""
import sys
import asyncio
import time
from unittest.mock import MagicMock

# Mock pyttsx3 before importing main to prevent ImportErrors in environments without libespeak
# and to allow mocking the blocking behavior.
mock_pyttsx3 = MagicMock()
sys.modules["pyttsx3"] = mock_pyttsx3

# Now import main
from backend.main import synthesize_speech, app, GROQ_API_KEY
import os

# Configure mock to simulate work
mock_engine = MagicMock()
mock_pyttsx3.init.return_value = mock_engine

def side_effect_run_and_wait():
    time.sleep(1.0) # Simulate blocking work

mock_engine.runAndWait.side_effect = side_effect_run_and_wait
# Also mock save_to_file to do nothing
mock_engine.save_to_file.return_value = None

# We need to mock file reading too since the function reads a file
import builtins
original_open = builtins.open

def mock_open(file, mode='r', *args, **kwargs):
    if str(file).endswith('.wav'):
        m = MagicMock()
        m.__enter__.return_value.read.return_value = b"fake wav data"
        return m
    return original_open(file, mode, *args, **kwargs)

import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_synthesize_speech_blocking():

    # Background task to measure loop responsiveness
    loop_times = []
    async def heartbeat():
        while True:
            loop_times.append(time.time())
            await asyncio.sleep(0.1)

    # Start heartbeat
    task = asyncio.create_task(heartbeat())

    # Give heartbeat a chance to start
    await asyncio.sleep(0.1)

    from httpx import AsyncClient, ASGITransport

    # We need to ensure ai_voice endpoint calls synthesize_speech.
    # It calls synthesize_speech ONLY if GROQ_API_KEY is set and FORCE_MOCK_GROQ is not "1"?
    # Let's check the code:
    # if not GROQ_API_KEY or os.getenv("FORCE_MOCK_GROQ") == "1":
    #    reply = f"You said: {req.prompt}"
    # else:
    #    ...
    # audio = synthesize_speech(reply)

    # It calls synthesize_speech in BOTH cases.

    with patch("builtins.open", side_effect=mock_open):
        with patch("os.remove"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                response = await ac.post("/ai-voice", json={"prompt": "hello"})

                assert response.status_code == 200

    # Stop heartbeat
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    if len(loop_times) > 0:
        max_gap = 0
        for i in range(1, len(loop_times)):
            gap = loop_times[i] - loop_times[i-1]
            if gap > max_gap:
                max_gap = gap

        # Verify call was made
        assert mock_engine.runAndWait.called, "synthesize_speech was not called!"

        # Assert that the loop was NOT blocked significantly (thanks to asyncio.to_thread)
        assert max_gap < 0.5, f"Event loop WAS blocked! Max gap: {max_gap:.4f}s"
    else:
        pytest.fail("Heartbeat didn't run at all")
