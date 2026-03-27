## 2024-03-27 - Using synchronous Groq client blocks FastAPI event loop
**Learning:** Using a synchronous API client (like `Groq`) inside a FastAPI `async def` endpoint is a performance anti-pattern. It completely blocks the event loop, pausing background tasks (like heartbeats) and preventing the server from handling concurrent requests until the network operation completes.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` their network calls in `async def` FastAPI endpoints to maintain concurrency.
