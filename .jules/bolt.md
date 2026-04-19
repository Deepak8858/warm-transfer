
## 2024-05-18 - [FastAPI Event Loop Blocked by Synchronous Groq Client]
**Learning:** Using the synchronous `Groq` client inside `async def` endpoints in FastAPI blocks the asyncio event loop for the duration of the network call to the LLM. This causes other concurrent tasks (e.g., heartbeats, handling other requests) to stall entirely. Testing revealed that a 2-second synchronous call allows only 1 heartbeat (the initial one), whereas an asynchronous call allows 5 heartbeats.
**Action:** Always use the `AsyncGroq` client and `await` network calls in FastAPI endpoints to maintain event loop responsiveness.
