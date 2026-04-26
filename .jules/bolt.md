## 2024-06-25 - Synchronous I/O in FastAPI Event Loop
**Learning:** Using synchronous clients like `Groq` or synchronous database calls (like SQLite via `sqlite3` without wrapper) inside an `async def` FastAPI endpoint completely blocks the event loop, causing 0 concurrent operations during the I/O wait.
**Action:** Always use async client variants (e.g., `AsyncGroq`) and `await` network calls. For local blocking operations like SQLite or CPU-bound tasks, explicitly offload them to `asyncio.to_thread` to maintain event loop responsiveness.

## 2024-06-25 - sqlite3 concurrency with asyncio.to_thread
**Learning:** The `backend/persistence.py` implementation uses a global SQLite connection initialized with `check_same_thread=False` and a global threading Lock. Wrapping operations in `asyncio.to_thread` works safely in this specific architecture to unblock the main loop.
**Action:** Verify the database connection properties (`check_same_thread=False` and presence of threading locks) before assuming that offloading `sqlite3` operations to `asyncio.to_thread` is safe from thread-related crashes.
