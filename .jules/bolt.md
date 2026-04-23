## 2025-01-20 - Sync API Clients in FastAPI Event Loop
**Learning:** Using synchronous API clients (like `Groq`) inside FastAPI `async def` endpoints is a performance anti-pattern that blocks the entire asyncio event loop, preventing concurrent handling of other incoming requests.
**Action:** Always use asynchronous API clients (e.g., `AsyncGroq`) and `await` network calls inside FastAPI `async def` endpoints to maintain event loop responsiveness and concurrency.
