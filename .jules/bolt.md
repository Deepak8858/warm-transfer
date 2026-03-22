## 2024-05-15 - Fast API Async API Blocking
**Learning:** Using synchronous API clients (like `Groq`) inside FastAPI `async def` endpoints is a performance anti-pattern that completely blocks the event loop, causing the server to pause handling concurrent requests.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` network calls inside FastAPI asynchronous endpoints to maintain concurrency.
