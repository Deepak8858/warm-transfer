## 2024-05-24 - AsyncGroq blocks FastAPI event loop

**Learning:** Using synchronous `Groq` client inside an `async def` FastAPI endpoint blocks the event loop. Even though `await asyncio.to_thread` is used for TTS and SQLite persistence, the `groq_client.chat.completions.create` network call is completely synchronous, severely degrading concurrency when multiple transfers or AI voice requests happen simultaneously.

**Action:** Replace `from groq import Groq` with `from groq import AsyncGroq` and use `await groq_client.chat.completions.create` to ensure network requests are non-blocking within the FastAPI asyncio event loop.
