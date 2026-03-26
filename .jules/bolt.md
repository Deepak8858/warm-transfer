## 2024-05-24 - Fast API Synchronous Blockers
**Learning:** Synchronous database operations (like SQLite persistence calls) and synchronous API clients (like `Groq`) block the FastAPI event loop entirely, preventing concurrent request handling and causing missed heartbeats.
**Action:** Always offload synchronous operations to separate threads using `asyncio.to_thread` or use asynchronous counterparts (e.g., `AsyncGroq`) in FastAPI `async def` endpoints to maintain event loop responsiveness.
