## 2024-05-20 - FastAPI Event Loop Blocking
**Learning:** SQLite operations using the stdlib `sqlite3` are synchronous and block the Python thread. In a FastAPI application, executing these directly in `async def` endpoints blocks the entire asyncio event loop, severely degrading concurrent request handling performance.
**Action:** Always wrap synchronous database calls (like `persistence.create_transfer_record` or `persistence.list_transfers`) in `await asyncio.to_thread(...)` to offload them to a worker thread and keep the event loop unblocked.
