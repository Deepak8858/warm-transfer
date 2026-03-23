## 2024-03-24 - AsyncGroq blocks event loop
**Learning:** Using synchronous API clients (like `Groq`) inside FastAPI `async def` endpoints is a performance anti-pattern that blocks the event loop. Always use asynchronous clients (e.g., `AsyncGroq`) and `await` network calls to maintain concurrency.
**Action:** Replace `Groq` with `AsyncGroq` in FastAPI endpoints.
