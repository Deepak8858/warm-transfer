import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient

# Add backend to sys.path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create temp DB file
temp_db = tempfile.NamedTemporaryFile(delete=False)
temp_db.close()
os.environ["PERSIST_DB_PATH"] = temp_db.name

# Mock env vars
os.environ["LIVEKIT_API_KEY"] = "test"
os.environ["LIVEKIT_API_SECRET"] = "test"
os.environ["LIVEKIT_URL"] = "http://test"

from main import app, persistence

client = TestClient(app)

def setup_module(module):
    # Initialize DB
    persistence.init_db()

def teardown_module(module):
    if os.path.exists(temp_db.name):
        os.remove(temp_db.name)

def test_list_transfers_excludes_context():
    # 1. Create a transfer with a large context
    large_context = "A" * 1000
    resp = client.post("/initiate-transfer", json={
        "call_context": large_context,
        "room_name": "perf-test-room",
        "agent_a_identity": "perf-agent"
    })
    assert resp.status_code == 200
    data = resp.json()
    transfer_id = data["id"]

    # 2. List transfers
    list_resp = client.get("/transfers?limit=10")
    assert list_resp.status_code == 200
    transfers = list_resp.json()["transfers"]

    # Find our transfer
    target = next(t for t in transfers if t["id"] == transfer_id)

    # 3. Verify call_context
    ctx = target.get('call_context')
    print(f"List view call_context: {ctx}")
    assert ctx is None or ctx == "", "call_context should be empty in list view for performance"

    # 4. Verify detail view HAS the context
    detail_resp = client.get(f"/transfers/{transfer_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["call_context"] == large_context

if __name__ == "__main__":
    setup_module(None)
    try:
        test_list_transfers_excludes_context()
    finally:
        teardown_module(None)
    print("Test finished")
