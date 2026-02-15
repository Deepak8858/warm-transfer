import unittest
import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import persistence

class TestPersistence(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the database
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.test_dir.name) / "test.db"

        # Patch DB_PATH in persistence module
        self.original_db_path = persistence.DB_PATH
        persistence.DB_PATH = self.db_path

        # Initialize DB
        persistence.init_db()

    def tearDown(self):
        # Restore DB_PATH
        persistence.DB_PATH = self.original_db_path
        self.test_dir.cleanup()

    def test_list_transfers_excludes_call_context(self):
        # Create a transfer record
        rec_id = persistence.create_transfer_record(
            room_name="room1",
            agent_a="agent_a",
            summary="summary",
            call_context="context"
        )

        # List transfers
        transfers = persistence.list_transfers()

        self.assertEqual(len(transfers), 1)
        transfer = transfers[0]
        self.assertEqual(transfer['id'], rec_id)
        self.assertIsNone(transfer['call_context'])
        self.assertEqual(transfer['summary'], "summary")

    def test_get_transfer_includes_call_context(self):
        rec_id = persistence.create_transfer_record(
            room_name="room1",
            agent_a="agent_a",
            summary="summary",
            call_context="context"
        )

        transfer = persistence.get_transfer(rec_id)
        self.assertIsNotNone(transfer)
        self.assertEqual(transfer['call_context'], "context")

if __name__ == '__main__':
    unittest.main()
