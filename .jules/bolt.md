## 2024-06-25 - Prevent Event Loop Blocking with AsyncGroq
**Learning:** In a FastAPI application, using a synchronous HTTP client like `Groq` for network I/O requests within async endpoints blocks the main event loop, significantly degrading performance for concurrent requests.
**Action:** Always use the asynchronous equivalent (`AsyncGroq`) and `await` the network calls to ensure the event loop remains unblocked and responsive during LLM API operations.
