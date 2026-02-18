import unittest
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock persistence before import if needed, but we can just use env var
# We need to ensure persistence is reloaded or initialized with new DB path
import importlib

class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.test_dir.name) / "test.db"
        os.environ["PERSIST_DB_PATH"] = str(self.db_path)

        # Reload persistence to pick up env var
        if 'persistence' in sys.modules:
             import persistence
             importlib.reload(persistence)
        else:
             import persistence

        self.persistence = persistence
        self.persistence.init_db()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_create_and_list_transfers(self):
        # Create a transfer
        rec_id = self.persistence.create_transfer_record(
            room_name="room1",
            agent_a="agentA",
            summary="summary1",
            call_context="context1"
        )

        # List transfers
        transfers = self.persistence.list_transfers()
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0]["id"], rec_id)
        self.assertEqual(transfers[0]["summary"], "summary1")
        # Check that call_context is NOT present in list view (optimization)
        self.assertNotIn("call_context", transfers[0])

    def test_get_transfer(self):
        rec_id = self.persistence.create_transfer_record(
            room_name="room2",
            agent_a="agentA",
            summary="summary2",
            call_context="context2"
        )

        transfer = self.persistence.get_transfer(rec_id)
        self.assertIsNotNone(transfer)
        self.assertEqual(transfer["call_context"], "context2")

if __name__ == "__main__":
    unittest.main()
