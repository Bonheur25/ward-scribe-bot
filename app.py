from dotenv import load_dotenv
load_dotenv()
import os
import logging
import requests
import time
import threading
import mimetypes
from datetime import datetime
from flask import Flask, request, jsonify
from collections import deque

# =========================
# APP SETUP
# =========================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ward-scribe")

# =========================
# ENV VARIABLES
# =========================

HF_TOKEN = os.getenv("HF_TOKEN")

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
# HUGGINGFACE WHISPER
# =========================

def transcribe_hf(audio_bytes, content_type=None):

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set")

    # ✅ UPDATED WORKING HF ENDPOINT
    API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-small"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": content_type or "application/octet-stream"
    }

    response = requests.post(
        API_URL,
        headers=headers,
        data=audio_bytes,
        timeout=120
    )

    # DEBUG LOGS
    log.info(f"HF STATUS: {response.status_code}")
    log.info(f"HF RESPONSE: {response.text[:500]}")

    if response.status_code != 200:
        return f"HF ERROR {response.status_code}"

    result = response.json()

    if isinstance(result, dict) and "text" in result:
        return result["text"]

    if isinstance(result, list) and len(result) > 0:
        return result[0].get("text", "")

    return "No transcription available"

# =========================
# CLINICAL STRUCTURE
# =========================

def _extract_generated_text(result):
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("generated_text") or first.get("text") or ""

    if isinstance(result, dict):
        if result.get("error"):
            raise RuntimeError(result["error"])

        return (
            result.get("generated_text")
            or result.get("text")
            or result.get("summary_text")
            or ""
        )

    return ""


def structure_text(transcript):

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"

    prompt = f"""<s>[INST]
You are a careful clinical ward scribe. Convert the transcript into a concise SOAP-style ward note.

Rules:
- Use only information supported by the transcript.
- Do not invent demographics, vitals, exam findings, diagnoses, investigations, or treatments.
- If a field is not available, write "Not stated" for that field.
- Keep the note clinically useful and specific to the transcript.
- Return only the note in exactly this format:

PATIENT:
...

TIME:
{now}

TRANSCRIPT:
...

SUBJECTIVE:
...

ASSESSMENT:
...

PLAN:
- ...

DIFFERENTIALS:
- ...

PEARL:
...

Transcript:
{transcript}
[/INST]"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 700,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    log.info(f"HF STRUCTURE STATUS: {response.status_code}")
    log.info(f"HF STRUCTURE RESPONSE: {response.text[:500]}")

    if response.status_code != 200:
        raise RuntimeError(f"HF text generation error {response.status_code}: {response.text[:200]}")

    generated = _extract_generated_text(response.json()).strip()

    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()

    if generated:
        return generated

    return f"""
PATIENT:
Not specified

TIME:
{now}

TRANSCRIPT:
{transcript}

SUBJECTIVE:
{transcript}

ASSESSMENT:
Not stated

PLAN:
- Not stated

DIFFERENTIALS:
- Not stated

PEARL:
Not stated
"""

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

            transcript = transcribe_hf(audio, content_type)

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
