## 2024-05-24 - Network Call Optimization in remove_participant
**Learning:** LiveKit's Python SDK allows targeted removal using `api.RoomParticipantIdentity`, which throws an exception if the participant isn't found. This eliminates the need for an initial `list_participants` O(n) API fetch just to check existence.
**Action:** Always favor EAFP (Easier to Ask for Forgiveness than Permission) direct API actions with try-catch blocks over read-then-write patterns when dealing with external SDKs to save network round-trips.
