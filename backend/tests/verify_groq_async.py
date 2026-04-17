import asyncio
import time
from groq import AsyncGroq
import os

async def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping real call, no key")
        return
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    start = time.time()
    res = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hi"}],
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=10
    )
    print(f"Time: {time.time() - start:.2f}s, Res: {res.choices[0].message.content}")

if __name__ == "__main__":
    asyncio.run(main())
