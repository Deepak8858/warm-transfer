## 2024-10-25 - Redundant Network Calls in LiveKit API
**Learning:** Listing participants before removing a participant via the LiveKit API adds an unnecessary network round-trip, introducing 100-200ms of latency. The API supports direct removal by identity.
**Action:** Use direct entity operations (like `remove_participant` with `RoomParticipantIdentity`) instead of listing and filtering entities whenever the API supports it, saving redundant network requests.
