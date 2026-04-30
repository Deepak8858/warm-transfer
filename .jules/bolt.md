## 2024-05-24 - Unblocking the FastAPI Event Loop
**Learning:** Synchronous API clients (`Groq`) and synchronous database calls (SQLite) inside FastAPI `async def` endpoints block the main event loop, causing concurrency issues where other requests are stalled during long LLM generations.
**Action:** Always use `AsyncGroq` for network calls and `asyncio.to_thread` for SQLite persistence calls to keep the FastAPI event loop fully responsive.
