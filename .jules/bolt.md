## 2024-04-08 - Use AsyncGroq instead of Groq
**Learning:** Using synchronous API clients like `Groq` inside FastAPI `async def` endpoints is a performance anti-pattern that blocks the event loop.
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and `await` network calls to maintain concurrency.
