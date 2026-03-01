## 2024-05-20 - [Blocking FastAPI Event Loop]
**Learning:** Network calls like `Groq(...)` requests, and some DB operations (e.g. SQLite queries) were running synchronously in the async path, blocking the entire FastAPI event loop causing concurrent requests (like fetching tokens) to hang indefinitely until the inference was complete.
**Action:** Always prefer asynchronous equivalents like `AsyncGroq` for IO-bound blocking network calls in FastAPI routes. Use `asyncio.to_thread` for DB transactions if they must run synchronously but are fast enough not to require full async DB drivers.
