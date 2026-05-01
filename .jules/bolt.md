## 2024-05-24 - Unblocking FastAPI Event Loop
**Learning:** Using synchronous network clients (like `Groq`) or synchronous database operations (like SQLite queries) inside FastAPI `async def` endpoints blocks the main event loop, severely restricting concurrency and overall throughput.
**Action:** Replace synchronous clients with their async equivalents (e.g., `AsyncGroq`) and use `await`. For blocking local I/O like SQLite, offload the work using `asyncio.to_thread` to maintain a responsive event loop.
