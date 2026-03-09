## 2026-03-09 - Optimize Groq chat completion call to be asynchronous
**Learning:** The application was making a blocking synchronous network request to Groq via the `Groq` client inside an asynchronous FastAPI endpoint, blocking the event loop. This leads to poor concurrency.
**Action:** Replaced the synchronous `Groq` client with `AsyncGroq` from the `groq` package and used `await` for the `chat.completions.create` call to make the network request non-blocking.
