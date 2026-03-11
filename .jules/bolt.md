## 2024-03-20 - [Network Request Optimization in LiveKit Participant Removal]
**Learning:** Checking existence before performing an action (e.g., listing all participants to verify a participant is in the room before removing them) is a common anti-pattern that creates redundant O(N) network requests. LiveKit's direct removal API automatically handles cases where the participant does not exist.
**Action:** Always prefer direct mutation/action APIs instead of fetch-and-filter patterns, handling exceptions for missing resources appropriately to eliminate unnecessary network round-trips.
