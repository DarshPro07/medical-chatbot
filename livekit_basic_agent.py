"""
LiveKit Voice Agent - Groq LLM (openai/gpt-oss-120b) + ElevenLabs (Liam voice)
- LLM: Groq via OpenAI-compatible endpoint
- STT: ElevenLabs (if plugin provides STT; otherwise text-only input)
- TTS: ElevenLabs (defaults to Liam)

Required .env:
  GROQ_API_KEY=...
  GROQ_MODEL=openai/gpt-oss-120b
  GROQ_BASE_URL=https://api.groq.com/openai/v1

  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE=Liam                  # default; or set ELEVENLABS_VOICE_ID=<uuid>
  # Optional:
  # ELEVENLABS_TTS_MODEL=eleven_turbo_v2
  # ELEVENLABS_STT_MODEL=<model_name>

Run:
  python livekit_basic_agent.py console
"""

import os
import logging
from functools import lru_cache
from datetime import datetime
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool

# LLM client using OpenAI-compatible spec, pointed at Groq
from livekit.plugins import openai
from livekit.plugins import silero

# ElevenLabs plugin (required for TTS; STT may or may not be present)
try:
    from livekit.plugins import elevenlabs as lk_elevenlabs
except Exception as e:
    raise RuntimeError(
        f"livekit-plugins-elevenlabs not available: {e}. "
        "Install with: pip install -U livekit-plugins-elevenlabs elevenlabs"
    )

# Load environment
load_dotenv(".env")

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("livekit_basic_agent")


@lru_cache(maxsize=16)
def resolve_elevenlabs_voice_id(api_key: str, voice_name: str) -> str | None:
    """Resolve an ElevenLabs voice name (e.g., 'Liam') to voice_id once per process."""
    if not voice_name:
        return None
    try:
        from elevenlabs import ElevenLabs  # pip install elevenlabs
        client = ElevenLabs(api_key=api_key)
        resp = client.voices.get_all()
        voices = getattr(resp, "voices", []) or []
        for v in voices:
            if getattr(v, "name", "").lower() == voice_name.lower():
                vid = getattr(v, "voice_id", None) or getattr(v, "id", None)
                if vid:
                    logger.info(f"Resolved ElevenLabs voice '{voice_name}' to voice_id '{vid}'.")
                    return vid
        logger.warning(f"ElevenLabs voice '{voice_name}' not found; using default voice.")
    except Exception as e:
        logger.warning(f"Could not resolve ElevenLabs voice '{voice_name}': {e}. Using default voice.")
    return None


class Assistant(Agent):
    """Basic voice assistant with Airbnb booking capabilities."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful and friendly Airbnb voice assistant. "
                "You can help users search for Airbnbs in different cities and book their stays. "
                "Keep your responses concise and natural, as if having a conversation."
            )
        )

        # Mock Airbnb database
        self.airbnbs = {
            "san francisco": [
                {"id": "sf001", "name": "Cozy Downtown Loft", "address": "123 Market Street, San Francisco, CA", "price": 150, "amenities": ["WiFi", "Kitchen", "Workspace"]},
                {"id": "sf002", "name": "Victorian House with Bay Views", "address": "456 Castro Street, San Francisco, CA", "price": 220, "amenities": ["WiFi", "Parking", "Washer/Dryer", "Bay Views"]},
                {"id": "sf003", "name": "Modern Studio near Golden Gate", "address": "789 Presidio Avenue, San Francisco, CA", "price": 180, "amenities": ["WiFi", "Kitchen", "Pet Friendly"]},
            ],
            "new york": [
                {"id": "ny001", "name": "Brooklyn Brownstone Apartment", "address": "321 Bedford Avenue, Brooklyn, NY", "price": 175, "amenities": ["WiFi", "Kitchen", "Backyard Access"]},
                {"id": "ny002", "name": "Manhattan Skyline Penthouse", "address": "555 Fifth Avenue, Manhattan, NY", "price": 350, "amenities": ["WiFi", "Gym", "Doorman", "City Views"]},
                {"id": "ny003", "name": "Artsy East Village Loft", "address": "88 Avenue A, Manhattan, NY", "price": 195, "amenities": ["WiFi", "Washer/Dryer", "Exposed Brick"]},
            ],
            "los angeles": [
                {"id": "la001", "name": "Venice Beach Bungalow", "address": "234 Ocean Front Walk, Venice, CA", "price": 200, "amenities": ["WiFi", "Beach Access", "Patio"]},
                {"id": "la002", "name": "Hollywood Hills Villa", "address": "777 Mulholland Drive, Los Angeles, CA", "price": 400, "amenities": ["WiFi", "Pool", "City Views", "Hot Tub"]},
            ],
        }

        # Track bookings
        self.bookings = []

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    @function_tool
    async def search_airbnbs(self, context: RunContext, city: str) -> str:
        """Search for available Airbnbs in a city."""
        city_lower = city.lower()
        if city_lower not in self.airbnbs:
            return (
                f"Sorry, I don't have any Airbnb listings for {city} at the moment. "
                "Available cities are: San Francisco, New York, and Los Angeles."
            )

        listings = self.airbnbs[city_lower]
        lines = [f"Found {len(listings)} Airbnbs in {city}:\n"]
        for listing in listings:
            lines += [
                f"• {listing['name']}",
                f"  Address: {listing['address']}",
                f"  Price: ${listing['price']} per night",
                f"  Amenities: {', '.join(listing['amenities'])}",
                f"  ID: {listing['id']}\n",
            ]
        return "\n".join(lines)

    @function_tool
    async def book_airbnb(self, context: RunContext, airbnb_id: str, guest_name: str, check_in_date: str, check_out_date: str) -> str:
        """Book an Airbnb."""
        # Find the Airbnb
        airbnb = None
        for city_listings in self.airbnbs.values():
            for listing in city_listings:
                if listing['id'] == airbnb_id:
                    airbnb = listing
                    break
            if airbnb:
                break

        if not airbnb:
            return f"Sorry, I couldn't find an Airbnb with ID {airbnb_id}. Please search for available listings first."

        # Create booking
        booking = {
            "confirmation_number": f"BK{len(self.bookings) + 1001}",
            "airbnb_name": airbnb['name'],
            "address": airbnb['address'],
            "guest_name": guest_name,
            "check_in": check_in_date,
            "check_out": check_out_date,
            "total_price": airbnb['price'],
        }
        self.bookings.append(booking)

        return (
            "✓ Booking confirmed!\n\n"
            f"Confirmation Number: {booking['confirmation_number']}\n"
            f"Property: {booking['airbnb_name']}\n"
            f"Address: {booking['address']}\n"
            f"Guest: {booking['guest_name']}\n"
            f"Check-in: {booking['check_in']}\n"
            f"Check-out: {booking['check_out']}\n"
            f"Nightly Rate: ${booking['total_price']}\n\n"
            "You'll receive a confirmation email shortly. Have a great stay!"
        )


async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent (console-friendly)."""

    # LLM: Groq via OpenAI-compatible endpoint using your chosen model
    groq_api = os.getenv("GROQ_API_KEY")
    if not groq_api:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    llm = openai.LLM(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=groq_api,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )

    # ElevenLabs TTS (Liam by default)
    el_api = os.getenv("ELEVENLABS_API_KEY")
    if not el_api:
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env")

    el_tts_model = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_turbo_v2")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
    voice_name = os.getenv("ELEVENLABS_VOICE", "Liam").strip()

    if not voice_id and voice_name:
        voice_id = resolve_elevenlabs_voice_id(el_api, voice_name) or ""

    if voice_id:
        tts = lk_elevenlabs.TTS(api_key=el_api, voice_id=voice_id, model=el_tts_model)
        logger.info(f"Using ElevenLabs TTS voice_id '{voice_id}' (model={el_tts_model}).")
    else:
        tts = lk_elevenlabs.TTS(api_key=el_api, model=el_tts_model)
        logger.info(f"Using ElevenLabs TTS default voice (model={el_tts_model}).")

    # ElevenLabs STT (optional)
    stt = None
    if hasattr(lk_elevenlabs, "STT"):
        stt_model = os.getenv("ELEVENLABS_STT_MODEL")
        stt_kwargs = {"api_key": el_api}
        if stt_model:
            stt_kwargs["model"] = stt_model
        try:
            stt = lk_elevenlabs.STT(**stt_kwargs)
            logger.info("ElevenLabs STT initialized.")
        except Exception as e:
            logger.warning(f"Failed to init ElevenLabs STT, continuing without STT (text-only input): {e}")
            stt = None
    else:
        logger.info("ElevenLabs plugin STT not found. Running in text-input mode (no mic).")

    # Configure the voice pipeline
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=silero.VAD.load(),
    )

    # Start the session
    await session.start(room=ctx.room, agent=Assistant())

    # Initial greeting
    await session.generate_reply(instructions="Greet the user warmly and ask how you can help.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))