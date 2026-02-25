## 2026-02-25 - [Blocking Persistence]
**Learning:** `backend/main.py` uses synchronous `sqlite3` calls directly in async route handlers, blocking the event loop.
**Action:** Use `asyncio.to_thread` for all persistence layer calls to unblock the main thread.
