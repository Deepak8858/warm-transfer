## 2024-11-20 - [Offload SQLite DB Calls]
**Learning:** In a FastAPI application using async/await, relying on standard synchronous SQLite operations under heavy concurrency blocks the event loop. This leads to starvation for other concurrent endpoints (like simple health checks).
**Action:** Always wrap synchronous disk or database operations, especially those using module-level locking in Python, with `asyncio.to_thread` to preserve FastAPI's concurrent handling capabilities.
