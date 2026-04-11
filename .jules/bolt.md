## 2024-05-24 - AsyncGroq usage
**Learning:** The FastAPI backend endpoints in `backend/main.py` (`/ai-voice` and `/initiate-transfer`) are defined as `async def` but are using the synchronous `Groq` client (`groq_client = Groq(...)` and `groq_client.chat.completions.create(...)`). This blocks the FastAPI event loop during external network calls.
**Action:** Use `AsyncGroq` from `groq` and `await` the API calls (`await groq_client.chat.completions.create(...)`) inside FastAPI `async def` endpoints to ensure concurrency and avoid blocking the event loop.
