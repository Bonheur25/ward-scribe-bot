from dotenv import load_dotenv
load_dotenv()
import os
import logging
import requests
import time
import threading
import mimetypes
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
from collections import deque
from urllib.parse import unquote, urlparse

# =========================
# APP SETUP
# =========================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ward-scribe")

# =========================
# ENV VARIABLES
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GREEN_API_ID = os.getenv("GREEN_API_ID")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN")

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

GREEN_BASE = f"https://api.green-api.com/waInstance{GREEN_API_ID}"

# =========================
# QUEUE
# =========================

task_queue = deque()
processed = set()
lock = threading.Semaphore(2)

# =========================
# WHATSAPP HELPERS
# =========================

def send_whatsapp(chat_id, msg):
    try:
        url = f"{GREEN_BASE}/sendMessage/{GREEN_API_TOKEN}"

        requests.post(
            url,
            json={
                "chatId": chat_id,
                "message": msg
            },
            timeout=15
        )

    except Exception as e:
        log.error(f"WhatsApp send error: {e}")


def download_audio(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
    return response.content, content_type


def guess_audio_content_type(audio_url=None, provided_content_type=None, file_name=None):
    if provided_content_type:
        return provided_content_type.split(";")[0].strip()

    if file_name:
        guessed_type, _ = mimetypes.guess_type(file_name)
        if guessed_type:
            return guessed_type

    if audio_url:
        guessed_type, _ = mimetypes.guess_type(audio_url.split("?", 1)[0])
        if guessed_type:
            return guessed_type

    return "application/octet-stream"

# =========================
# GROQ WHISPER
# =========================

GROQ_AUDIO_EXTENSIONS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}

GROQ_CONTENT_TYPE_EXTENSIONS = {
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/mp4a-latm": ".m4a",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


def _groq_audio_extension(audio_url=None, content_type=None, file_name=None):
    candidates = []

    if file_name:
        candidates.append(file_name)

    if audio_url:
        path = unquote(urlparse(audio_url).path)
        candidates.append(path)

    for candidate in candidates:
        _, extension = os.path.splitext(candidate.split("?", 1)[0])
        extension = extension.lower()
        if extension in GROQ_AUDIO_EXTENSIONS:
            return extension

    if content_type:
        clean_content_type = content_type.split(";", 1)[0].strip().lower()
        if clean_content_type in GROQ_CONTENT_TYPE_EXTENSIONS:
            return GROQ_CONTENT_TYPE_EXTENSIONS[clean_content_type]

        guessed_extension = mimetypes.guess_extension(clean_content_type)
        if guessed_extension:
            guessed_extension = guessed_extension.lower()
            if guessed_extension in GROQ_AUDIO_EXTENSIONS:
                return guessed_extension

    return ".ogg"


def transcribe_groq(audio_bytes, content_type=None, file_name=None, audio_url=None):

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    data = {
        "model": "whisper-large-v3",
        "prompt": (
            "Uyu murwayi, yahageze, nimugoroba, ejo, bamushyize, "
            "turacyagereje, gusa, twanamusabize, muri, bed, labs, fluids, "
            "ciplo, orthopedie, ceftriaxone, Ringers Lactate, antibiotique, "
            "oxygen therapy, steroid, immunosuppressant, medecine interne, "
            "sodium, electrolyte, cardiac arrest, ambulance, consultation, "
            "rendez vous, ibintu, kugenda, murwayi, indwara"
        )
    }

    extension = _groq_audio_extension(
        audio_url=audio_url,
        content_type=content_type,
        file_name=file_name
    )
    upload_name = f"audio{extension}"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        with open(temp_path, "rb") as audio_file:
            files = {
                "file": (
                    upload_name,
                    audio_file,
                    content_type or mimetypes.types_map.get(extension, "application/octet-stream")
                )
            }

            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError as e:
                log.warning(f"Could not remove temp audio file {temp_path}: {e}")

    log.info(f"GROQ TRANSCRIPTION STATUS: {response.status_code}")
    log.info(f"GROQ TRANSCRIPTION RESPONSE: {response.text[:500]}")

    if response.status_code != 200:
        raise RuntimeError(f"Groq transcription error {response.status_code}: {response.text[:200]}")

    result = response.json()

    if isinstance(result, dict) and "text" in result:
        return result["text"]

    return "No transcription available"

# =========================
# CLINICAL STRUCTURE
# =========================

def structure_text(transcript):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    system_prompt = (
        "You are a clinical ward scribe in Rwanda. Doctors speak in mixed "
        "Kinyarwanda, French and English. Never say Not stated - always infer "
        "clinically from context. If something is unclear flag it with a "
        "warning emoji. Output structured ward note with: Patient, Admission, "
        "Presenting complaint, Management so far, Pending, Plan, Clinical flags."
    )

    user_prompt = f"""
Create a structured ward note in English from the transcript below.

Transcript:
{transcript}
""".strip()

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.2,
            "top_p": 0.9
        },
        timeout=120
    )

    log.info(f"GROQ WARD NOTE STATUS: {response.status_code}")
    log.info(f"GROQ WARD NOTE RESPONSE: {response.text[:500]}")

    if response.status_code != 200:
        raise RuntimeError(f"Groq ward note generation error {response.status_code}: {response.text[:200]}")

    result = response.json()
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"Groq ward note generation error: {result['error']}")

    try:
        generated = result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Groq chat completion response: {str(result)[:300]}") from e

    if generated:
        return generated

    raise RuntimeError("Groq ward note generation returned empty content")

# =========================
# NOTION SAVE
# =========================

def save_to_notion(content, msg_id):

    try:
        from notion_client import Client

        notion = Client(auth=NOTION_TOKEN)

        notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},

            properties={
                "Name": {
                    "title": [{
                        "text": {
                            "content": f"Ward Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    }]
                },

                "Source": {
                    "rich_text": [{
                        "text": {
                            "content": msg_id
                        }
                    }]
                }
            },

            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": content[:1900]
                            }
                        }]
                    }
                }
            ]
        )

    except Exception as e:
        log.error(f"Notion save error: {e}")

# =========================
# WORKER
# =========================

def worker():

    while True:

        if not task_queue:
            time.sleep(1)
            continue

        task = task_queue.popleft()

        chat_id = task["chat_id"]
        msg_id = task["msg_id"]
        audio_url = task["audio_url"]
        audio_content_type = task.get("audio_content_type")
        file_name = task.get("file_name")

        if msg_id in processed:
            continue

        processed.add(msg_id)

        try:

            send_whatsapp(chat_id, "Processing audio...")

            audio, downloaded_content_type = download_audio(audio_url)

            content_type = guess_audio_content_type(
                audio_url=audio_url,
                provided_content_type=audio_content_type or downloaded_content_type,
                file_name=file_name
            )

            transcript = transcribe_groq(audio, content_type, file_name, audio_url)

            result = structure_text(transcript)

            save_to_notion(result, msg_id)

            send_whatsapp(
                chat_id,
                "Saved note:\n\n" + result[:1500]
            )

        except Exception as e:

            log.error(f"Worker error: {e}")

            send_whatsapp(
                chat_id,
                f"Error processing audio: {str(e)[:100]}"
            )

# START WORKER
threading.Thread(target=worker, daemon=True).start()

# =========================
# WEBHOOK
# =========================

@app.route("/webhook", methods=["POST"])
def webhook():

    data = request.get_json(force=True)

    if not data:
        return jsonify({"status": "no data"}), 200

    msg_type = data.get("messageData", {}).get("typeMessage", "")

    chat_id = data.get("senderData", {}).get("chatId", "")

    msg_id = data.get("idMessage", "")

    if msg_type in ["audioMessage", "voiceMessage", "pttMessage"]:

        file_data = data.get("messageData", {}).get("fileMessageData", {})

        audio_url = file_data.get("downloadUrl")
        audio_content_type = (
            file_data.get("mimeType")
            or file_data.get("mimetype")
            or file_data.get("contentType")
            or file_data.get("content-type")
        )
        file_name = file_data.get("fileName") or file_data.get("filename")

        if not audio_url:
            audio_url = f"{GREEN_BASE}/downloadFile/{GREEN_API_TOKEN}/{msg_id}"

        task_queue.append({
            "chat_id": chat_id,
            "msg_id": msg_id,
            "audio_url": audio_url,
            "audio_content_type": audio_content_type,
            "file_name": file_name
        })

    return jsonify({"status": "queued"}), 200

# =========================
# HEALTH CHECK
# =========================

@app.route("/", methods=["GET"])
def health():
    return "Ward Scribe HF Whisper Running"

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
