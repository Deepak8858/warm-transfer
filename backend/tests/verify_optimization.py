
import sys
import os
import unittest
import uuid
import time
from pathlib import Path

# Set up path to import backend modules
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

import persistence

class TestOptimization(unittest.TestCase):
    def setUp(self):
        # Use a temporary DB for testing
        self.db_path = f"test_db_{uuid.uuid4()}.db"
        # Monkey patch persistence.DB_PATH
        persistence.DB_PATH = Path(self.db_path)
        persistence.init_db()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_list_excludes_context(self):
        # Create a record with a large context
        large_context = "x" * 10000
        rec_id = persistence.create_transfer_record(
            room_name="room1",
            agent_a="agentA",
            summary="summary",
            call_context=large_context
        )

        # List transfers
        transfers = persistence.list_transfers()
        self.assertTrue(len(transfers) >= 1)

        target_record = next((t for t in transfers if t['id'] == rec_id), None)
        self.assertIsNotNone(target_record)

        # Verify call_context is NOT in the list dictionary
        self.assertNotIn('call_context', target_record)
        print("\n✅ Verified: call_context is excluded from list_transfers")

        # Verify get_transfer DOES return context
        full_record = persistence.get_transfer(rec_id)
        self.assertIsNotNone(full_record)
        self.assertIn('call_context', full_record)
        self.assertEqual(full_record['call_context'], large_context)
        print("✅ Verified: call_context is preserved in get_transfer")

if __name__ == '__main__':
    unittest.main()
