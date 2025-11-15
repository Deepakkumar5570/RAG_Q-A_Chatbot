import speech_recognition as sr
import tempfile
from io import BytesIO
from pydub import AudioSegment

def transcribe_audio(audio_bytes):
    """Convert recorded audio to text using SpeechRecognition"""
    recognizer = sr.Recognizer()

    # Convert byte data to WAV for recognition
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")

    with sr.AudioFile(temp_wav.name) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn’t understand that."
    except sr.RequestError:
        return "Speech Recognition service unavailable."
