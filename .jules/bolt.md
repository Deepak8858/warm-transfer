## 2024-05-24 - Unnecessary Network Round-Trip in LiveKit Participant Removal
**Learning:** Calling `list_participants` to verify a participant's existence before calling `remove_participant` adds an unnecessary network round-trip. `remove_participant` handles non-existent participants gracefully or throws an exception which can be caught, meaning the initial fetch is completely redundant.
**Action:** Always call `remove_participant` directly and rely on error handling to manage cases where the participant is already gone, saving ~100-200ms of latency per transfer.
