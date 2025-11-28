# tools/audio_transcribing.py

from langchain_core.tools import tool
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client()


@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file (MP3/WAV) into text using Gemini.

    The file is expected to be located inside the LLMFiles directory,
    as returned by the download_file tool.

    Args:
        file_path (str): The filename returned by download_file (e.g. "audio.mp3").

    Returns:
        str: The transcribed text from the audio, or an error message.
    """
    try:
        full_path = os.path.join("LLMFiles", file_path)

        if not os.path.exists(full_path):
            return f"Error: file not found at {full_path}"

        # Guess MIME type from extension
        ext = os.path.splitext(full_path)[1].lower()
        if ext == ".wav":
            mime = "audio/wav"
        elif ext == ".mp3":
            mime = "audio/mpeg"
        else:
            mime = "audio/*"

        with open(full_path, "rb") as f:
            audio_bytes = f.read()

        # Gemini transcription: ask for plain text transcript
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                types.Part.from_data(
                    data=audio_bytes,
                    mime_type=mime,
                ),
                "Transcribe the spoken content to plain text with all numbers written as digits."
            ],
        )

        text = (response.text or "").strip()
        if not text:
            return "Error: empty transcript from Gemini"

        return text
    except Exception as e:
        return f"Error occurred while transcribing: {e}"
