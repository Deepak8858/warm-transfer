## 2025-05-05 - Avoid synchronous persistence blocking FastAPI event loop
**Learning:** Found multiple synchronous persistence methods (`persistence.create_transfer_record`, `persistence.set_agent_b`, `persistence.list_transfers`, `persistence.get_transfer`) in the FastAPI backend main module. Synchronous operations executed on the main event loop will block async routing for the entire server during those operations.
**Action:** Always wrap synchronous persistence calls inside the `asyncio.to_thread` coroutine to prevent blocking the FastAPI asynchronous event loop, improving overall concurrent handling performance for the API application.

## 2025-05-05 - Use async groq client for performance improvement
**Learning:** Found synchronous execution of Groq client inference within asynchronous FastAPI controllers. Using synchronous Groq inference causes thread blocking and limits concurrency while waiting for inference completions.
**Action:** Replace `from groq import Groq` with `from groq import AsyncGroq` and initialize `AsyncGroq(api_key=GROQ_API_KEY)` when dealing with completions. Await asynchronous methods (`await groq_client.chat.completions.create`) to ensure we do not block the event loop for inference API calls.
