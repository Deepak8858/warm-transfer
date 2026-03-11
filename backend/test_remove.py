import asyncio
from livekit import api
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    lk_api = api.LiveKitAPI()
    try:
        await lk_api.room.remove_participant(
            api.RoomParticipantIdentity(
                room="nonexistent_room",
                identity="nonexistent_identity"
            )
        )
        print("Success?")
    except Exception as e:
        print(f"Failed with exception: {type(e)} - {e}")

if __name__ == "__main__":
    asyncio.run(main())
