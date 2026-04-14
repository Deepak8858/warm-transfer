import asyncio
import time
from backend.main import initiate_transfer, TransferRequest

async def test_performance():
    # Setup test
    req = TransferRequest(
        call_context="User is calling to check their balance. Authenticated.",
        room_name="test-room-123",
        agent_a_identity="agent-a"
    )

    # Measure time taken
    start = time.time()
    await initiate_transfer(req)
    duration = time.time() - start

    print(f"Time taken: {duration:.3f} seconds")

asyncio.run(test_performance())
