
## 2024-05-19 - Unblocking the FastAPI Event Loop
**Learning:** Using synchronous HTTP clients (`Groq()`) or performing synchronous file/database I/O (`persistence.*`) directly inside `async def` endpoints completely blocks the underlying event loop, destroying the concurrency benefits of FastAPI. This codebase relied on synchronous DB wrappers behind an `asyncio.to_thread` for the `pyttsx3` engine but missed other blocking calls.
**Action:** When working in FastAPI, actively convert all I/O bound synchronous libraries (like standard `requests` or `Groq()`) to their asynchronous equivalents (`AsyncGroq()`, `httpx.AsyncClient`). Any remaining synchronous operations (like `sqlite3`) must be explicitly wrapped in `asyncio.to_thread()` when called from within an `async def` context.
