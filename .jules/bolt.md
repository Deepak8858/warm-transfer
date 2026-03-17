## 2024-05-14 - Blocking event loop via db operations
**Learning:** Database operations on SQLite block the event loop in `backend/main.py`.
**Action:** Use `asyncio.to_thread` for SQLite DB persistence operations.
