from pathlib import Path

import requests
from django.shortcuts import render


FASTAPI_URL = "http://127.0.0.1:8000/transcribe"


def transcript_ui(request):
    context = {
        "transcript": None,
        "error": None,
        "filename": None,
    }

    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        if not audio_file:
            context["error"] = "Please select an audio file."
            return render(
                request,
                "t_app/transcript.html",
                context,
            )

        allowed_extensions = {
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

        extension = Path(audio_file.name).suffix.lower()

        if extension not in allowed_extensions:
            context["error"] = (
                f"Unsupported file type: {extension}"
            )

            return render(
                request,
                "t_app/transcript.html",
                context,
            )

        try:
            response = requests.post(
                FASTAPI_URL,
                files={
                    "audio": (
                        audio_file.name,
                        audio_file.file,
                        audio_file.content_type,
                    )
                },
                timeout=300,
            )
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_message = error_data.get(
                        "detail",
                        "Transcription failed.",
                    )
                except Exception:
                    error_message = (
                        "Transcription service returned an error."
                    )

                context["error"] = error_message

                return render(
                    request,
                    "t_app/transcript.html",
                    context,
                )

            data = response.json()

            context["filename"] = data.get(
                "filename",
                audio_file.name,
            )

            context["transcript"] = data.get(
                "transcript",
                "",
            )

        except requests.exceptions.ConnectionError:

            context["error"] = (
                "Could not connect to the Speech-to-Text API. "
                "Make sure FastAPI is running on port 8000."
            )

        except requests.exceptions.Timeout:

            context["error"] = (
                "Transcription took too long and timed out."
            )

        except Exception as exc:

            context["error"] = (
                f"Unexpected error: {exc}"
            )

    return render(
        request,
        "transcript.html",
        context,
    )
