import asyncio
import time
from groq import Groq

def test_blocking():
    client = Groq(api_key="sk-test", max_retries=0)
    try:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model="llama3-8b-8192"
        )
    except Exception as e:
        pass

async def heartbeat():
    ticks = 0
    start = time.time()
    while time.time() - start < 1:
        await asyncio.sleep(0.1)
        ticks += 1
    return ticks

async def main():
    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.1) # give it time to start
    start = time.time()
    # Mocking block
    test_blocking()
    # Simulate a delay if Groq is fast or fails fast
    time.sleep(0.5)
    ticks = await task
    print(f"Ticks with sync blocking: {ticks}")

asyncio.run(main())
