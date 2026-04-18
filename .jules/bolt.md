## 2024-04-18 - Groq Async Client Optimization
**Learning:** Using the synchronous `Groq` client within FastAPI `async def` endpoints (e.g. `initiate_transfer`, `ai_voice`) is a performance anti-pattern that blocks the event loop. The app utilizes a global `groq_client`, which means concurrent asynchronous requests wait sequentially for external network calls if the synchronous client is used.
**Action:** Replace `Groq` with `AsyncGroq` from `groq` to prevent event loop blocking when making AI completions, matching best practices for FastAPI.
