from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from validator import AudioAwareValidator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "input.mp3"

GEMINI_MODEL_NAME = "gemini-2.5-flash"

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
# Speech-to-text using Groq Whisper
# ---------------------------------------------------------

def transcribe_audio(
    audio_path: Path,
) -> str:

    print("=" * 60)
    print("Transcribing audio using Groq Whisper...")
    print("=" * 60)

    try:

        client = Groq(
            api_key=GROQ_API_KEY
        )

        print("Groq client initialized.")

        with audio_path.open("rb") as file:

            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                temperature=0,
                response_format="verbose_json",
            )

        transcript = transcription.text.strip()

        if not transcript:
            raise RuntimeError(
                "Whisper returned an empty transcript."
            )

        return transcript

    except Exception as exc:

        raise RuntimeError(
            f"Groq Whisper transcription failed: {exc}"
        ) from exc


# ---------------------------------------------------------
# Validate transcript against original audio
# ---------------------------------------------------------

def validate_transcript(
    audio_path: Path,
    transcript: str,
) -> dict:

    print()
    print("=" * 60)
    print("Validating transcript against original audio")
    print("=" * 60)

    validator = AudioAwareValidator(
        model_name=GEMINI_MODEL_NAME,
    )

    return validator.validate(
        audio_path=audio_path,
        transcript=transcript,
    )


# ---------------------------------------------------------
# Apply corrections
# ---------------------------------------------------------

def apply_corrections(
    transcript: str,
    validation_results: dict,
) -> str:

    updated_transcript = transcript

    discrepancies = validation_results.get(
        "discrepancies",
        [],
    )

    for discrepancy in discrepancies:

        original = discrepancy.get(
            "transcript_says",
            "",
        )

        corrected = discrepancy.get(
            "audio_actually_says",
            "",
        )

        if not original:
            continue

        updated_transcript = updated_transcript.replace(
            original,
            corrected,
            1,
        )

    return updated_transcript


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    print(f"Audio: {AUDIO_PATH}")

    # -----------------------------------------------------
    # Step 1: Speech-to-text
    # -----------------------------------------------------

    transcript = transcribe_audio(
        AUDIO_PATH
    )

    print()
    print("=" * 60)
    print("WHISPER TRANSCRIPT")
    print("=" * 60)
    print(transcript)

    # -----------------------------------------------------
    # Step 2: Audio-aware validation
    # -----------------------------------------------------

    validation_results = validate_transcript(
        audio_path=AUDIO_PATH,
        transcript=transcript,
    )

    print()
    print("=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(
        json.dumps(
            validation_results,
            indent=2,
            ensure_ascii=False,
        )
    )

    # -----------------------------------------------------
    # Step 3: Apply corrections
    # -----------------------------------------------------

    updated_transcript = apply_corrections(
        transcript=transcript,
        validation_results=validation_results,
    )

    # -----------------------------------------------------
    # Step 4: Final output
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL TRANSCRIPT")
    print("=" * 60)

    print(updated_transcript)

    output = {
        "audio_file": str(AUDIO_PATH),
        "original_transcript": transcript,
        "validation": validation_results,
        "updated_transcript": updated_transcript,
    }

    output_path = BASE_DIR / "validation_result.json"

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
