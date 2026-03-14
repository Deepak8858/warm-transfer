## 2024-05-24 - Async API Clients and DB Offloading
**Learning:** Using synchronous API clients (like `Groq`) or SQLite operations inside FastAPI `async def` endpoints completely blocks the event loop, causing requests to queue up and destroying concurrency.
**Action:** Always use asynchronous versions of clients (e.g., `AsyncGroq`) and wrap blocking operations like SQLite writes in `asyncio.to_thread()` when operating in an asynchronous web framework like FastAPI.
