## 2024-05-24 - Offload synchronous database calls to thread pool
**Learning:** In FastAPI, synchronous operations like SQLite database calls block the asynchronous event loop, which can heavily degrade performance and pause data transmission for concurrent requests.
**Action:** Always wrap synchronous blocking operations (such as `sqlite3` persistence calls or local file operations) with `asyncio.to_thread` to offload these calls to separate threads and prevent blocking the FastAPI event loop. Keyword arguments can be passed directly to `asyncio.to_thread`.
