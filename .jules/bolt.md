## 2025-01-26 - Blocking TTS Operations
**Learning:** Python's `pyttsx3` library has blocking methods like `runAndWait()` which halt the main thread. In an `async` FastAPI endpoint, calling this directly blocks the entire event loop, serializing all requests.
**Action:** Always wrap blocking CPU/IO-bound synchronous calls (like local TTS or heavy data processing) in `await asyncio.to_thread(...)` within async route handlers to maintain concurrency.
