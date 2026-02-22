## 2025-02-23 - Async Persistence Bottleneck
**Learning:** `sqlite3` operations in Python are blocking by default. Calling them directly in `async def` FastAPI routes freezes the entire event loop, causing requests to be processed sequentially even with multiple clients.
**Action:** Always wrap blocking I/O (like `sqlite3` or `pyttsx3`) in `asyncio.to_thread()` when working within async frameworks like FastAPI. Also, verify concurrency with a simple reproduction script that simulates delay.
