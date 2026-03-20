## 2024-03-XX - Missing offload to thread in main.py
**Learning:** SQLite persistence calls block the FastAPI event loop because they run synchronously.
**Action:** Use `asyncio.to_thread` for all direct SQLite calls in FastAPI `async def` endpoints.
