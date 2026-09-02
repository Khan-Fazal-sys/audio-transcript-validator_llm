from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import Groq
from pydantic import BaseModel
import uvicorn


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Please add GROQ_API_KEY to your .env file."
    )


# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("=" * 60)
print("Groq client initialized.")
print("Whisper model: whisper-large-v3-turbo")
print("=" * 60)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="Speech-to-Text API",
    version="1.0.0",
    description="API for audio transcription and transcript validation",
)


# ---------------------------------------------------------
# Response model
# ---------------------------------------------------------

class TranscriptResponse(BaseModel):
    filename: str
    transcript: str
    status: str


# ---------------------------------------------------------
# Supported audio formats
# ---------------------------------------------------------

ALLOWED_TYPES = {
    # MP3
    "audio/mpeg",
    "audio/mp3",

    # WAV
    "audio/wav",
    "audio/x-wav",
    "audio/wave",

    # FLAC
    "audio/flac",
    "audio/x-flac",

    # M4A / MP4
    "audio/mp4",
    "audio/x-m4a",

    # OGG
    "audio/ogg",
    "application/ogg",

    # WebM
    "audio/webm",

    # AAC
    "audio/aac",

    # AIFF
    "audio/aiff",
    "audio/x-aiff",
}


ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".mp4",
    ".ogg",
    ".webm",
    ".aac",
    ".aiff",
}


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
async def root() -> dict[str, str]:

    return {
        "message": "Speech-to-Text API is running"
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# Transcription
# ---------------------------------------------------------

@app.post(
    "/transcribe",
    response_model=TranscriptResponse,
)
async def transcribe(
    audio: UploadFile = File(...),
) -> TranscriptResponse:

    original_filename = audio.filename or "audio"

    extension = Path(
        original_filename
    ).suffix.lower()


    # -----------------------------------------------------
    # Validate extension
    # -----------------------------------------------------

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio extension: {extension}. "
                f"Supported formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


    # -----------------------------------------------------
    # Validate MIME type
    # -----------------------------------------------------

    if (
        audio.content_type
        and audio.content_type not in ALLOWED_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio MIME type: "
                f"{audio.content_type}"
            ),
        )


    # -----------------------------------------------------
    # Generate safe filename
    # -----------------------------------------------------

    filename = f"{uuid4()}{extension}"

    file_path = UPLOAD_DIR / filename


    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    try:

        with file_path.open("wb") as buffer:

            while chunk := await audio.read(
                1024 * 1024
            ):
                buffer.write(chunk)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded audio.",
        ) from exc

    finally:

        await audio.close()


    # -----------------------------------------------------
    # Transcribe using Groq Whisper
    # -----------------------------------------------------

    try:

        print()
        print("=" * 60)
        print("Transcribing audio with Groq Whisper...")
        print(f"File: {file_path}")
        print("=" * 60)

        with file_path.open("rb") as file:

            transcription = (
                groq_client
                .audio
                .transcriptions
                .create(
                    file=file,
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    response_format="verbose_json",
                )
            )


        transcript = transcription.text.strip()


        if not transcript:

            raise RuntimeError(
                "Whisper returned an empty transcript."
            )


        print()
        print("=" * 60)
        print("TRANSCRIPT")
        print("=" * 60)
        print(transcript)


    except Exception as exc:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return TranscriptResponse(
        filename=original_filename,
        transcript=transcript,
        status="success",
    )


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "api_script:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
