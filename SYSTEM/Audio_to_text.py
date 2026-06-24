import json
import wave
import os
from vosk import Model, KaldiRecognizer
# Step 1: Download a model (one-time) from https://alphacephei.com/vosk/models
# Recommended: vosk-model-small-en-us-0.15 (~40MB)
# Unzip it so you have a folder like: ./vosk-model-small-en-us-0.15
MODEL_PATH = "vosk-model-small-en-us-0.15"
name = "input_audio.wav"  # must be mono, 16-bit PCM WAV
def transcribe_audio(name ,file_path = "SYSTEM/Data", model_path = ""):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Download it first.")
    audio_path = os.path.join(file_path,name)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at '{audio_path}'. Check the path/filename.")
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Audio must be mono, 16-bit PCM WAV. Convert it using ffmpeg first.")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    full_text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            full_text.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    full_text.append(final_result.get("text", ""))
    return " ".join(t for t in full_text if t).strip()
if __name__ == "__main__":
    transcript = transcribe_audio(name , MODEL_PATH)
    print(transcript)
    