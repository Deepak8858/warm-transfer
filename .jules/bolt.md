## 2024-04-06 - Async HTTP Clients in FastAPI endpoints
**Learning:** Using synchronous HTTP clients like `Groq` in an `async def` FastAPI endpoint blocks the event loop. The event loop cannot process other requests while waiting for the LLM's response, creating a severe bottleneck under concurrent loads.
**Action:** When implementing new dependencies or integrations in FastAPI `async` handlers, ensure they are natively asynchronous (e.g., `AsyncGroq`, `httpx.AsyncClient`) or wrap blocking operations with `asyncio.to_thread`.
