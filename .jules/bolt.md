## 2024-05-04 - Unblock FastAPI Event Loop with Async API Clients
**Learning:** Using synchronous API clients (like the default `Groq` client) in FastAPI endpoints running on the asyncio event loop causes the entire loop to block during network I/O, freezing concurrent requests.
**Action:** Always prefer asynchronous versions of SDK clients (e.g., `AsyncGroq` instead of `Groq`) in asynchronous web frameworks. If an async client is unavailable, use `asyncio.to_thread` for the synchronous calls.
