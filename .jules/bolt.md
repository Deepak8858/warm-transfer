## 2024-05-14 - Prevent FastAPI Event Loop Blocking

**Learning:** The FastAPI backend event loop was being blocked by two types of synchronous operations: `Groq` API client calls and SQLite operations in `backend/main.py` (`persistence.py`). Fast event loop execution is critical for concurrent request handling and maintaining low latency for things like `StreamingResponse` objects used in `ai_voice`.

**Action:** Replaced `Groq` with `AsyncGroq` for non-blocking network I/O. Wrapped synchronous `persistence` module calls (`create_transfer_record`, `set_agent_b`, `list_transfers`, `get_transfer`) with `asyncio.to_thread`. Used `functools.partial` to safely pass keyword arguments to `asyncio.to_thread` to maintain compatibility with older Python versions.
