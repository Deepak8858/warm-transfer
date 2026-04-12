## 2024-05-24 - Unblocking the FastAPI Event Loop
**Learning:** Using synchronous clients like `Groq` and making synchronous SQLite calls directly in FastAPI `async def` endpoints blocks the main event loop, causing severe latency under load.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq` with `await`) for network requests and wrap synchronous disk/DB operations in `asyncio.to_thread` within `async def` endpoints.
