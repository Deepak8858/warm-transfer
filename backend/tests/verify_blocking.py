import asyncio
import time

async def heartbeat(stop_event, counts):
    while not stop_event.is_set():
        counts['ticks'] += 1
        await asyncio.sleep(0.1)

def blocking_work():
    time.sleep(1)
    return "done"

async def async_work():
    await asyncio.sleep(1)
    return "done"

async def main():
    print("Testing blocking work...")
    stop_event = asyncio.Event()
    counts = {'ticks': 0}
    task = asyncio.create_task(heartbeat(stop_event, counts))

    # Run blocking work (simulating synchronous Groq/SQLite)
    blocking_work()

    stop_event.set()
    await task
    print(f"Ticks during blocking work: {counts['ticks']}")

    print("Testing async/offloaded work...")
    stop_event = asyncio.Event()
    counts = {'ticks': 0}
    task = asyncio.create_task(heartbeat(stop_event, counts))

    # Run async work (simulating AsyncGroq / asyncio.to_thread)
    await async_work()

    stop_event.set()
    await task
    print(f"Ticks during async work: {counts['ticks']}")

if __name__ == "__main__":
    asyncio.run(main())
