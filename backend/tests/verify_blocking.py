import asyncio
import time
import os
import sys

# Append backend path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

async def heartbeat():
    ticks = 0
    start = time.time()
    while time.time() - start < 2:
        await asyncio.sleep(0.1)
        ticks += 1
    return ticks

async def test_blocking():
    # We will simulate a blocking call
    import time
    def sync_block():
        time.sleep(1)
        return "done"

    print("Testing sync blocking...")
    task = asyncio.create_task(heartbeat())
    sync_block()
    ticks = await task
    print(f"Sync ticks: {ticks}")

    print("Testing to_thread offloading...")
    task = asyncio.create_task(heartbeat())
    await asyncio.to_thread(sync_block)
    ticks = await task
    print(f"Async ticks: {ticks}")

if __name__ == "__main__":
    asyncio.run(test_blocking())
