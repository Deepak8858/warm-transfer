## 2024-05-24 - Async API Clients in FastAPI
**Learning:** Using synchronous external API clients (like `Groq`) inside FastAPI `async def` endpoints completely blocks the event loop, preventing concurrent requests from being handled.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` their network calls in `async def` handlers to maintain concurrency and responsiveness.
