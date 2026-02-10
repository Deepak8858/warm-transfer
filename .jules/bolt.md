## 2026-02-10 - [Backend] pyttsx3 Thread Safety
**Learning:** `pyttsx3` is not thread-safe. Calls via `asyncio.to_thread` must be serialized using an `asyncio.Lock` to prevent concurrent access issues.
**Action:** When optimizing blocking TTS operations, wrap `to_thread` with a global `asyncio.Lock`.
