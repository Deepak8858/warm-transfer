## 2024-05-24 - Async Offloading for SQLite Persistence
**Learning:** Synchronous database operations in `backend/main.py` block the event loop entirely, whereas offloading them to `asyncio.to_thread` maintains responsiveness. The `sqlite3.connect` is initialized with `check_same_thread=False` and uses a global lock, which safely allows offloading to a thread.
**Action:** Always wrap direct SQLite persistence calls (like `create_transfer_record`, `list_transfers`, `get_transfer`, and `set_agent_b`) in `asyncio.to_thread()` within FastAPI async endpoints to prevent blocking the event loop.
