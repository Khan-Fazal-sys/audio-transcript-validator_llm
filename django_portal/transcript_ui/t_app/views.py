import uuid
from pathlib import Path

import requests
from django.shortcuts import render, redirect

FASTAPI_URL = "http://127.0.0.1:8000/transcribe"

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".webm", ".aac", ".aiff",
}

# how the transcript is keyed in session — swap for a DB row + id if you
# expect concurrent users, since session storage is per-browser only
SESSION_KEY_PREFIX = "transcript_result_"


def transcript_ui(request):
    """GET: render an empty upload form.
    POST: validate + call the FastAPI service, then redirect to a GET
    URL that shows the result. Refreshing that GET url just re-fetches
    the same stored result instead of re-submitting the audio file.
    """
    if request.method == "POST":
        return _handle_upload(request)

    return render(request, "transcript.html", {
        "transcript": None,
        "error": None,
        "filename": None,
    })


def _handle_upload(request):
    audio_file = request.FILES.get("audio")

    if not audio_file:
        return _render_error(request, "Please select an audio file.")

    extension = Path(audio_file.name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return _render_error(request, f"Unsupported file type: {extension}")

    try:
        response = requests.post(
            FASTAPI_URL,
            files={"audio": (audio_file.name, audio_file.file, audio_file.content_type)},
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        return _render_error(
            request,
            "Could not connect to the Speech-to-Text API. "
            "Make sure FastAPI is running on port 8000.",
        )
    except requests.exceptions.Timeout:
        return _render_error(request, "Transcription took too long and timed out.")
    except Exception as exc:
        return _render_error(request, f"Unexpected error: {exc}")

    if response.status_code != 200:
        try:
            error_message = response.json().get("detail", "Transcription failed.")
        except Exception:
            error_message = "Transcription service returned an error."
        return _render_error(request, error_message)

    data = response.json()

    # stash result server-side and redirect — this is what fixes the
    # stale-transcript-on-refresh problem
    result_id = uuid.uuid4().hex
    request.session[SESSION_KEY_PREFIX + result_id] = {
        "transcript": data.get("transcript", ""),
        "filename": data.get("filename", audio_file.name),
    }
    return redirect("transcript_result", result_id=result_id)


def transcript_result(request, result_id):
    """Plain GET page for a completed transcript. Refreshing this URL
    re-reads the same stored result instead of replaying the upload.
    Result is popped on read, so a second refresh shows a clean form —
    change to .get() instead of .pop() if you'd rather it persist.
    """
    data = request.session.pop(SESSION_KEY_PREFIX + result_id, None)
    if not data:
        return redirect("transcript_ui")

    return render(request, "transcript.html", {
        "transcript": data["transcript"],
        "error": None,
        "filename": data["filename"],
    })


def _render_error(request, message):
    return render(request, "transcript.html", {
        "transcript": None,
        "error": message,
        "filename": None,
    })