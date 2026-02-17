import sys
import os
import unittest
import tempfile
import shutil
import sqlite3
from unittest.mock import MagicMock

# Add backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set env var for DB path BEFORE importing persistence
tmp_dir = tempfile.mkdtemp()
db_path = os.path.join(tmp_dir, "test_perf.db")
os.environ["PERSIST_DB_PATH"] = db_path

# Now import persistence (no external deps)
import persistence

class TestPerfListTransfers(unittest.TestCase):
    def setUp(self):
        # persistence.init_db() uses the DB_PATH we set
        persistence.init_db()
        # Create a record with a large context
        self.large_context = "A" * 10000
        self.rec_id = persistence.create_transfer_record(
            room_name="room1",
            agent_a="agentA",
            summary="summary",
            call_context=self.large_context
        )

    def tearDown(self):
        shutil.rmtree(tmp_dir)

    def test_list_transfers_behavior(self):
        transfers = persistence.list_transfers()
        self.assertTrue(len(transfers) > 0)
        t = transfers[0]

        # Optimization: call_context should NOT be in the listed items
        self.assertNotIn("call_context", t)
        print("call_context is CORRECTLY missing from list_transfers")

    def test_get_transfer_includes_context(self):
        t = persistence.get_transfer(self.rec_id)
        self.assertIsNotNone(t)
        self.assertIn("call_context", t)
        self.assertEqual(t["call_context"], self.large_context)
        print("call_context is CORRECTLY present in get_transfer")

if __name__ == "__main__":
    unittest.main()
