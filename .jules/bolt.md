## 2025-01-26 - Blocking TTS Operations in FastAPI
**Learning:** The `pyttsx3` library used for Text-to-Speech performs synchronous, blocking I/O operations (file writing and external process execution) which block the main asyncio event loop in FastAPI applications. This causes the entire server to become unresponsive during speech synthesis. Furthermore, `pyttsx3` is not thread-safe, so offloading it to threads via `asyncio.to_thread` requires serialization using an `asyncio.Lock`.
**Action:** Always offload CPU-bound or blocking I/O tasks to a thread pool using `asyncio.to_thread` in async endpoints. When using non-thread-safe libraries like `pyttsx3` in a threaded context, ensure proper locking mechanisms are in place to prevent race conditions or crashes.

## 2025-01-26 - Testing with Missing System Dependencies
**Learning:** The development environment lacks `libespeak.so.1`, causing `pyttsx3` imports to fail with `OSError`.
**Action:** When testing code that depends on system libraries not present in the environment, use `unittest.mock.MagicMock` to mock the module in `sys.modules` *before* importing the application code. This allows verifying logic and performance characteristics without requiring the actual binary dependencies.
