## 2024-05-24 - AsyncGroq and offloaded persistence in FastAPI
**Learning:** Using synchronous API clients (`Groq`) or local blocking disk/DB calls inside FastAPI `async def` endpoints blocks the event loop, causing concurrency regressions (0 heartbeats).
**Action:** Always use asynchronous clients (e.g., `AsyncGroq`) and offload local synchronous work (e.g., SQLite via `persistence` module) using `asyncio.to_thread` to maintain loop responsiveness.
