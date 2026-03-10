## 2024-05-24 - AsyncGroq Non-Blocking Implementation
**Learning:** Using the synchronous `Groq` client blocks the FastAPI event loop during LLM API calls, decreasing throughput. Even though we correctly used `asyncio.to_thread` for SQLite DB calls, we missed that `Groq` makes blocking network requests that can natively use `async`/`await`.
**Action:** Always prefer `AsyncGroq` over `Groq` in an async web framework like FastAPI to ensure non-blocking I/O operations and maximize concurrency without needing thread pools.
