## 2024-03-24 - Asyncio SQLite Persistence Optimization
**Learning:** Synchronous SQLite database operations in `backend/main.py` (like `persistence.create_transfer_record`, `persistence.list_transfers`) block the FastAPI event loop, causing poor responsiveness during concurrent requests.
**Action:** Always offload synchronous database persistence calls to separate threads using `asyncio.to_thread` in a FastAPI architecture that relies on `async def` endpoints, to ensure the main event loop remains unblocked.
