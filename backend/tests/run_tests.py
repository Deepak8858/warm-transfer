import os
import sys
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Mock pyttsx3 before importing main app as it requires libespeak which is missing
mock_pyttsx3 = MagicMock()
sys.modules["pyttsx3"] = mock_pyttsx3

# Ensure env present for app startup
os.environ.setdefault("LIVEKIT_API_KEY", "test_key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test_secret")
os.environ.setdefault("LIVEKIT_URL", "wss://example.com")
os.environ.setdefault("FORCE_MOCK_GROQ", "1")
os.environ.setdefault("ENABLE_AGENT_MOCK", "1")

load_dotenv()

from main import app  # noqa: E402


def test_initiate_and_list_transfers():
    client = TestClient(app)
    r = client.post("/initiate-transfer", json={
        "call_context": "Customer asks about billing.",
        "room_name": "test-room",
        "agent_a_identity": "Agent A",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert "summary" in data
    assert data.get("id")

    # list transfers
    r2 = client.get("/transfers?limit=5")
    assert r2.status_code == 200, r2.text
    listed = r2.json().get("transfers", [])
    assert any(t.get("id") == data["id"] for t in listed)


def test_ai_voice_mock():
    client = TestClient(app)
    r = client.post("/ai-voice", json={"prompt": "Say hello"})
    assert r.status_code == 200, r.text
    # Should be audio/wav
    assert r.headers.get("content-type", "").startswith("audio/wav")


def test_agent_endpoints_mock():
    client = TestClient(app)
    room = "test-room-agent"
    # start
    r1 = client.post("/agent/start", json={"room_name": room, "identity": "ai-agent"})
    assert r1.status_code == 200, r1.text
    # say
    r2 = client.post("/agent/say", json={"room_name": room, "text": "Hello customer"})
    assert r2.status_code == 200, r2.text
    # stop
    r3 = client.post("/agent/stop", json={"room_name": room})
    assert r3.status_code == 200, r3.text


if __name__ == "__main__":
    # Simple runner
    try:
        test_initiate_and_list_transfers()
        test_ai_voice_mock()
        print("All tests passed.")
    except AssertionError as e:
        print("Test failed:", e)
        raise