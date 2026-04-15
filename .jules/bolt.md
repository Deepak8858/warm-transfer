## 2024-05-15 - [Initial Journal]
**Learning:** Initial journal created.
**Action:** Keep adding notes as we find performance issues.

## 2024-05-15 - [Preventing Event Loop Blocking with Synchronous I/O]
**Learning:** Calling synchronous database operations (like SQLite queries) or file I/O directly in FastAPI `async def` endpoints blocks the event loop, preventing the application from handling concurrent requests.
**Action:** Always wrap synchronous persistence calls and I/O operations with `await asyncio.to_thread(...)` in asynchronous endpoints.
