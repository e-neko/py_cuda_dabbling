from builtins import print
import os
import subprocess
from faster_whisper import WhisperModel
import yt_dlp
import requests
import sys

# 1. Configuration
VIDEO_URL = "https://youtube.com" + "/watch?v=jQscp-2qeTg"
AUDIO_OUTPUT = "downloaded_audio.mp3"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434") + "/api/generate"
MODEL_NAME = "llama3" # or llama3.2, mistral, phi3, etc.

# 1.5 Grab video url from command line argument if provided
if len(sys.argv) > 1:
    VIDEO_URL = sys.argv[1]
    print(f"Using video URL from command line: {VIDEO_URL}")


# 2. Download and extract audio using yt-dlp
print("📥 Downloading audio from YouTube...")
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'downloaded_audio.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'quiet': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([VIDEO_URL])

# 3. Transcribe using faster-whisper on GPU
print("🎙️ Transcribing audio (using GPU)...")
# Note: device="cuda" targets your NVIDIA GPU, compute_type="float16" optimizes VRAM usage
model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe(AUDIO_OUTPUT, beam_size=5)

transcript = ""
for segment in segments:
    transcript += f" {segment.text}"

print(f"\n📝 Transcription complete ({info.language_probability*100:.1f}% confidence in {info.language}).")

# 4. Send to Ollama for summary
print(f"🤖 Sending transcript to Ollama ({MODEL_NAME})...")
prompt = (
    "Please provide a concise bullet-point summary of the following transcript. "
    "Focus on the main ideas and actionable takeaways:\n\n"
    f"{transcript}"
)

try:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    summary = response.json().get("response", "")
    
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")

except requests.exceptions.RequestException as e:
    print(f"❌ Could not connect to Ollama. Ensure Ollama is running and accessible.")
    print(f"Error details: {e}")

# Clean up audio file
if os.path.exists(AUDIO_OUTPUT):
    os.remove(AUDIO_OUTPUT)
