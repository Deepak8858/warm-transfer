from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os
import logging
import time
import asyncio
from groq import Groq
from livekit import api
from contextlib import asynccontextmanager
try:
    # When backend is treated as a package (e.g. uvicorn backend.main:app)
    from . import persistence as persistence  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for running "python main.py" directly inside backend folder
    import persistence  # type: ignore

from fastapi.responses import StreamingResponse
import io
import pyttsx3
try:
    # local import for agent runtime manager
    from .agent_runtime import manager as agent_manager  # type: ignore
except Exception:
    from agent_runtime import manager as agent_manager  # type: ignore

# Load environment variables (.env resolved relative to this file, not CWD)
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # fallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events with runtime env validation."""
    logger.info("Starting Warm Transfer API")

    # (Re)load env vars at runtime so missing vars give clear runtime error
    global LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL, GROQ_API_KEY
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    missing = [k for k,v in {
        'LIVEKIT_API_KEY': LIVEKIT_API_KEY,
        'LIVEKIT_API_SECRET': LIVEKIT_API_SECRET,
        'LIVEKIT_URL': LIVEKIT_URL,
    }.items() if not v]
    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Environment variables validated")

    # Initialize persistence
    try:
        persistence.init_db()
    except Exception as e:
        logger.warning(f"Could not initialize persistence: {e}")

    yield

    # Shutdown cleanup
    global livekit_api
    if livekit_api:
        try:
            await livekit_api.aclose()
            logger.info("Closed LiveKit API client")
        except Exception as e:
            logger.warning(f"Error closing LiveKit API client: {e}")

def synthesize_speech(text: str) -> bytes:
    """Synthesize speech locally using pyttsx3 (offline).

    Returns WAV bytes. This is a simple local TTS fallback; in production you may
    want a cloud TTS for higher quality and latency guarantees.
    """
    engine = pyttsx3.init()
    # Configure voice/rate optionally
    engine.setProperty('rate', 175)
    buf = io.BytesIO()
    # pyttsx3 doesn't write to BytesIO directly; we can write to a temp file then read
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        temp_path = tf.name
    try:
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        with open(temp_path, 'rb') as f:
            data = f.read()
        return data
    finally:
        try:
            _os.remove(temp_path)
        except Exception:
            pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="Warm Transfer API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholders; real values loaded in lifespan for clearer error handling
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
groq_client = None  # Will be initialized when needed
livekit_api = None  # Will be initialized when needed
tts_lock = asyncio.Lock()  # Serialize pyttsx3 calls

def get_livekit_api():
    """Lazy-create and return LiveKit API client.

    Supports both newer and older constructor signatures.
    """
    global livekit_api, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    if livekit_api is None:
        try:
            # Newer SDK may accept api key & secret
            livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        except TypeError:
            livekit_api = api.LiveKitAPI(LIVEKIT_URL)
        logger.info("Initialized LiveKit API client")
    return livekit_api

def create_livekit_token(room_name: str, identity: str, name: str | None = None) -> str:
    """Create a LiveKit access token using the official SDK."""
    try:
        if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
            raise RuntimeError("LiveKit credentials not loaded at token creation time")
        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(name or identity)
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            ))
            .to_jwt()
        )
        return token
    except Exception as e:
        logger.error(f"Error creating LiveKit token: {e}")
        raise

async def create_room_if_not_exists(room_name: str):
    """
    Create a room if it doesn't already exist
    """
    try:
        lk_api = get_livekit_api()
        
        # Try to create the room
        room_info = await lk_api.room.create_room(
            api.CreateRoomRequest(name=room_name)
        )
        logger.info(f"Created room: {room_name}")
        return room_info
        
    except Exception as e:
        # Room might already exist, which is fine
        logger.info(f"Room {room_name} may already exist or creation failed: {str(e)}")
        return None

async def list_rooms():
    """
    List all active rooms
    """
    try:
        lk_api = get_livekit_api()
        result = await lk_api.room.list_rooms(api.ListRoomsRequest())
        return result.rooms
    except Exception as e:
        logger.error(f"Error listing rooms: {str(e)}")
        return []

async def remove_participant_from_room(room_name: str, identity: str):
    """
    Remove a participant from a room (for transfer completion)
    """
    try:
        lk_api = get_livekit_api()
        # This requires finding the participant first
        participants = await lk_api.room.list_participants(
            api.ListParticipantsRequest(room=room_name)
        )
        
        # Find the participant by identity
        target_participant = None
        for participant in participants.participants:
            if participant.identity == identity:
                target_participant = participant
                break
        
        if target_participant:
            await lk_api.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=room_name,
                    identity=identity
                )
            )
            logger.info(f"Removed participant {identity} from room {room_name}")
            return True
        else:
            logger.warning(f"Participant {identity} not found in room {room_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error removing participant: {str(e)}")
        return False

# Pydantic models for request/response
class TokenRequest(BaseModel):
    room_name: str
    identity: str

class TokenResponse(BaseModel):
    accessToken: str

class RoomInfo(BaseModel):
    name: str
    sid: str
    num_participants: int
    creation_time: int

class RoomsResponse(BaseModel):
    rooms: list[RoomInfo]

class TransferRequest(BaseModel):
    call_context: str
    room_name: str | None = None
    agent_a_identity: str | None = None

class TransferResponse(BaseModel):
    summary: str
    id: str | None = None
    room_name: str | None = None
    agent_a_identity: str | None = None
    briefing_room_name: str | None = None

class ParticipantInfo(BaseModel):
    identity: str
    name: str | None = None
    metadata: str | None = None

class ParticipantsResponse(BaseModel):
    room: str
    participants: list[ParticipantInfo]

class CompleteTransferRequest(BaseModel):
    original_room_name: str
    agent_a_identity: str
    agent_b_identity: str
    transfer_id: str | None = None

class SuccessResponse(BaseModel):
    success: bool
    message: str

class TransferRecord(BaseModel):
    id: str
    room_name: str
    agent_a: str
    agent_b: str | None
    summary: str
    call_context: str | None
    created_at: float

class TransferListResponse(BaseModel):
    transfers: list[TransferRecord]

class VoiceRequest(BaseModel):
    prompt: str
    voice: str | None = None

class AgentStartRequest(BaseModel):
    room_name: str
    identity: str | None = "ai-agent"

class AgentSayRequest(BaseModel):
    room_name: str
    text: str

class AgentStopRequest(BaseModel):
    room_name: str

@app.post("/ai-voice")
async def ai_voice(req: VoiceRequest):
    """Generate an AI response with Groq and synthesize to speech (WAV)."""
    try:
        # If Groq not configured, just echo
        if not GROQ_API_KEY or os.getenv("FORCE_MOCK_GROQ") == "1":
            reply = f"You said: {req.prompt}"
        else:
            global groq_client
            if groq_client is None:
                groq_client = Groq(api_key=GROQ_API_KEY)
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful realtime voice assistant."},
                    {"role": "user", "content": req.prompt},
                ],
                model="llama3-8b-8192",
                temperature=0.6,
                max_tokens=200,
            )
            reply = chat_completion.choices[0].message.content
        async with tts_lock:
            audio = await asyncio.to_thread(synthesize_speech, reply)
        return StreamingResponse(io.BytesIO(audio), media_type="audio/wav")
    except Exception as e:
        logger.error(f"AI voice error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate AI voice")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Warm Transfer API is running"}

@app.post("/agent/start")
async def agent_start(req: AgentStartRequest) -> SuccessResponse:
    try:
        # Ensure room exists to align with expected UX
        await create_room_if_not_exists(req.room_name)
        await agent_manager.start_agent(req.room_name, req.identity or "ai-agent")
        return SuccessResponse(success=True, message=f"Agent started for room {req.room_name}")
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to start agent")

@app.post("/agent/say")
async def agent_say(req: AgentSayRequest) -> SuccessResponse:
    try:
        await agent_manager.say(req.room_name, req.text)
        return SuccessResponse(success=True, message="Agent spoke")
    except Exception as e:
        logger.error(f"Failed agent say: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agent/stop")
async def agent_stop(req: AgentStopRequest) -> SuccessResponse:
    try:
        await agent_manager.stop_agent(req.room_name)
        return SuccessResponse(success=True, message=f"Agent stopped for room {req.room_name}")
    except Exception as e:
        logger.error(f"Failed to stop agent: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop agent")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/get-token")
async def get_token(room_name: str, identity: str) -> TokenResponse:
    """
    Generate a LiveKit access token for a client and ensure the room exists.
    
    Args:
        room_name: The name of the room to join
        identity: The identity/name of the participant
    
    Returns:
        TokenResponse containing the access token
    """
    try:
        # Ensure room exists
        await create_room_if_not_exists(room_name)
        
        # Create access token using LiveKit SDK
        access_token = create_livekit_token(room_name, identity)
        
        logger.info(f"Generated LiveKit token for {identity} in room {room_name}")
        
        return TokenResponse(accessToken=access_token)
    
    except Exception as e:
        logger.error(f"Error generating token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate access token")

@app.get("/rooms")
async def get_rooms() -> RoomsResponse:
    """
    List all active LiveKit rooms.
    
    Returns:
        RoomsResponse containing list of active rooms
    """
    try:
        rooms = await list_rooms()
        
        room_list = []
        for room in rooms:
            room_list.append(RoomInfo(
                name=room.name,
                sid=room.sid,
                num_participants=room.num_participants,
                creation_time=room.creation_time
            ))
        
        logger.info(f"Listed {len(room_list)} active rooms")
        return RoomsResponse(rooms=room_list)
    
    except Exception as e:
        logger.error(f"Error listing rooms: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list rooms")

@app.get("/participants")
async def get_participants(room_name: str) -> ParticipantsResponse:
    """List participants in a given room.

    Args:
        room_name: target room

    Returns:
        ParticipantsResponse with participant identities.
    """
    try:
        lk_api = get_livekit_api()
        participants = await lk_api.room.list_participants(
            api.ListParticipantsRequest(room=room_name)
        )
        infos: list[ParticipantInfo] = []
        for p in participants.participants:
            infos.append(ParticipantInfo(identity=p.identity, name=p.name, metadata=getattr(p, 'metadata', None)))
        return ParticipantsResponse(room=room_name, participants=infos)
    except Exception as e:
        logger.warning(f"Error fetching participants for {room_name}: {e}")
        # Return empty list rather than 500 to keep UI resilient
        return ParticipantsResponse(room=room_name, participants=[])

@app.post("/initiate-transfer")
async def initiate_transfer(request: TransferRequest) -> TransferResponse:
    """
    Start the transfer process by generating a call summary using Groq LLM.
    
    Args:
        request: TransferRequest containing call_context
    
    Returns:
        TransferResponse containing the generated summary
    """
    try:
        # Determine briefing room name (private room for Agent A <-> Agent B)
        briefing_room_name: str | None = None
        if request.room_name:
            import random, string
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            briefing_room_name = f"{request.room_name}-brief-{suffix}"

        # Try to create briefing room (best effort)
        try:
            if briefing_room_name:
                await create_room_if_not_exists(briefing_room_name)
        except Exception as e:
            logger.info(f"Could not create briefing room {briefing_room_name}: {e}")

        # If no key, or explicit mock flag, return mock immediately
        if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here" or os.getenv("FORCE_MOCK_GROQ") == "1":
            summary = (
                "Mock Summary: "
                f"{request.call_context[:120]}"[:120] +
                "... (Configure GROQ_API_KEY for real AI summary)"
            )
            logger.info("Returning mock summary (Groq not configured or forced)")
            # Offload blocking DB call to thread to prevent blocking event loop
            rec_id = await asyncio.to_thread(
                persistence.create_transfer_record,
                room_name=request.room_name or "unknown",
                agent_a=request.agent_a_identity or "unknown",
                summary=summary,
                call_context=request.call_context
            )
            return TransferResponse(
                summary=summary,
                id=rec_id,
                room_name=request.room_name,
                agent_a_identity=request.agent_a_identity,
                briefing_room_name=briefing_room_name,
            )

        # Attempt real Groq call
        global groq_client
        if groq_client is None:
            groq_client = Groq(api_key=GROQ_API_KEY)

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that creates concise summaries of customer service calls. "
                            "Include the customer's main issue, progress made, and remaining actions. Under 200 words."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize this call context for a warm transfer: {request.call_context}"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=300,
            )
            summary = chat_completion.choices[0].message.content
            logger.info("Generated Groq summary for transfer")
            rec_id = await asyncio.to_thread(
                persistence.create_transfer_record,
                room_name=request.room_name or "unknown",
                agent_a=request.agent_a_identity or "unknown",
                summary=summary,
                call_context=request.call_context
            )
            return TransferResponse(
                summary=summary,
                id=rec_id,
                room_name=request.room_name,
                agent_a_identity=request.agent_a_identity,
                briefing_room_name=briefing_room_name,
            )
        except Exception as groq_err:
            logger.warning(f"Groq call failed, falling back to mock summary: {groq_err}")
            fallback = (
                "Mock Summary (fallback due to Groq error): "
                f"{request.call_context[:120]}"[:120] + "..."
            )
            rec_id = await asyncio.to_thread(
                persistence.create_transfer_record,
                room_name=request.room_name or "unknown",
                agent_a=request.agent_a_identity or "unknown",
                summary=fallback,
                call_context=request.call_context
            )
            return TransferResponse(summary=fallback, id=rec_id, room_name=request.room_name, agent_a_identity=request.agent_a_identity)
    except Exception as e:
        logger.error(f"Unexpected error generating summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate call summary (unexpected)")

@app.post("/complete-transfer")
async def complete_transfer(request: CompleteTransferRequest) -> SuccessResponse:
    """
    Complete the transfer by removing Agent A from the original room.
    
    Args:
        request: CompleteTransferRequest with room and agent details
    
    Returns:
        SuccessResponse confirming the transfer completion
    """
    try:
        # Remove Agent A from the original room using LiveKit API
        success = await remove_participant_from_room(
            request.original_room_name, 
            request.agent_a_identity
        )
        
        if success:
            # Optionally update agent_b for persisted record
            if request.transfer_id:
                try:
                    updated = await asyncio.to_thread(
                        persistence.set_agent_b, request.transfer_id, request.agent_b_identity
                    )
                    if updated:
                        logger.info(f"Updated transfer {request.transfer_id} with agent_b {request.agent_b_identity}")
                except Exception as e:
                    logger.warning(f"Failed to set agent_b on transfer {request.transfer_id}: {e}")
            logger.info(f"Successfully removed {request.agent_a_identity} from room {request.original_room_name}")
            return SuccessResponse(
                success=True, 
                message=f"Transfer completed. {request.agent_a_identity} removed from {request.original_room_name}"
            )
        else:
            logger.warning(f"Could not remove {request.agent_a_identity} from room {request.original_room_name}")
            return SuccessResponse(
                success=False,
                message=f"Transfer signal sent, but could not automatically remove participant. Manual disconnect may be required."
            )
    
    except Exception as e:
        logger.error(f"Error completing transfer: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to complete transfer")

@app.get("/transfers")
async def list_transfers(room_name: str | None = None, limit: int = 50) -> TransferListResponse:
    try:
        rows = await asyncio.to_thread(persistence.list_transfers, room_name=room_name, limit=limit)
        return TransferListResponse(transfers=[TransferRecord(**r) for r in rows])
    except Exception as e:
        logger.error(f"Error listing transfers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list transfers")

@app.get("/transfers/{transfer_id}")
async def get_transfer(transfer_id: str) -> TransferRecord:
    rec = await asyncio.to_thread(persistence.get_transfer, transfer_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Transfer not found")
    return TransferRecord(**rec)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)