## 2024-04-17 - FastAPI Async Event Loop Blocking
**Learning:** Synchronous network calls, specifically using the standard `Groq` client instead of `AsyncGroq`, block the entire asyncio event loop in FastAPI `async def` endpoints, significantly degrading concurrent performance.
**Action:** Always use the asynchronous equivalent of clients (e.g., `AsyncGroq`) in FastAPI when performing network I/O to maintain event loop responsiveness.
