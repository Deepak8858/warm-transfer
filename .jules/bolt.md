## 2024-05-18 - [Async Groq Client]
**Learning:** Using synchronous API clients like `Groq` inside FastAPI `async def` endpoints blocks the event loop, causing severe performance degradation for concurrent requests.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` network calls to maintain concurrency.
