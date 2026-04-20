## 2024-05-24 - Async API Clients in FastAPI
**Learning:** Using synchronous API clients (like `Groq`) inside FastAPI `async def` endpoints blocks the event loop, causing severe latency and concurrency issues under load.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` network calls in FastAPI async endpoints to maintain concurrency.
