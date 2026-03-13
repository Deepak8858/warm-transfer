## 2024-03-13 - Redundant Network Round-Trip in LiveKit Participant Removal
**Learning:** Calling `list_participants` before `remove_participant` to verify presence is a performance anti-pattern that introduces a redundant network round-trip to LiveKit Cloud.
**Action:** Directly call `remove_participant` with `RoomParticipantIdentity` and rely on exception handling if the participant doesn't exist, cutting network overhead by 50% for this operation.

## 2024-03-13 - FastAPI Event Loop Blocked by Synchronous SQLite Calls
**Learning:** SQLite persistence calls (`create_transfer_record`, `list_transfers`, etc.) in `backend/main.py` are synchronous and block the FastAPI event loop entirely. Benchmarking confirmed that these operations reduce responsiveness (0 heartbeats detected during execution).
**Action:** Always offload synchronous database operations to `asyncio.to_thread` in this codebase to maintain FastAPI event loop responsiveness.

## 2024-03-13 - FastAPI Event Loop Blocked by Synchronous API Clients
**Learning:** Initializing and using synchronous API clients like `Groq` inside FastAPI `async def` endpoints is a performance anti-pattern. This blocks the entire event loop during network requests to external services, causing latency spikes and unresponsiveness for concurrent requests.
**Action:** Always use the asynchronous version of API clients (e.g., `AsyncGroq`) and `await` their responses in FastAPI endpoints to maintain event loop responsiveness during network IO.
