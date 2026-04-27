## 2024-05-24 - Async Clients in FastAPI
**Learning:** Using synchronous clients (like `Groq`) inside FastAPI `async def` endpoints blocks the event loop because the sync IO operation suspends the loop completely, stopping other requests from being handled concurrently.
**Action:** Always use asynchronous equivalents (`AsyncGroq`, `httpx.AsyncClient`) inside `async def` routing functions or offload the sync call to another thread (e.g., `asyncio.to_thread`).
