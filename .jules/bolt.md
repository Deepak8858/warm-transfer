## 2024-03-03 - SQLite blocking event loop in FastAPI
**Learning:** SQLite database interaction through standard DB drivers (like sqlite3) or when wrapped with a `threading.Lock` will synchronously block the FastAPI event loop during read/write operations, pausing everything else (including concurrent requests or StreamingResponses).
**Action:** Always offload synchronous/blocking persistence calls to `asyncio.to_thread` to maintain concurrency and throughput, even for lightweight or in-memory setups like SQLite.
