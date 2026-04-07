## 2024-05-15 - FastAPI Async Event Loop Blocking
**Learning:** Using synchronous API clients (like `Groq`) and synchronous disk/DB calls directly inside FastAPI `async def` endpoints completely blocks the event loop, pausing all concurrent request handling.
**Action:** Always use async variants (e.g., `AsyncGroq`) for external API calls and offload synchronous blocking work (like SQLite DB calls) to thread pools using `await asyncio.to_thread(...)`.
