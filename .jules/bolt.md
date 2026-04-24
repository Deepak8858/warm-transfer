## 2024-10-24 - Async API Client Anti-pattern
**Learning:** Using the synchronous `Groq` client inside FastAPI's `async def` endpoints (e.g., `/ai-voice`, `/initiate-transfer`) completely blocks the event loop during network requests, severely impacting concurrent performance and throughput.
**Action:** Always use the asynchronous client (`AsyncGroq`) and `await` network calls in FastAPI endpoints to maintain event loop responsiveness.
