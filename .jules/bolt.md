## 2024-05-18 - Non-blocking Groq API Client
**Learning:** Using synchronous API clients like `Groq` inside FastAPI `async def` endpoints will completely block the event loop while waiting for network I/O.
**Action:** Always use the asynchronous versions of SDK clients (e.g., `AsyncGroq` instead of `Groq`) and `await` network calls inside FastAPI to maintain concurrency.
