"""
SELFY BAR — Backend AI per riconoscimento bottiglie
Riceve una foto in base64, la analizza con AI e restituisce le bottiglie trovate.
"""

import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# ---------- CONFIGURAZIONE ----------
# Supporta sia OpenAI (GPT-4o) che Anthropic (Claude)
# Imposta UNA delle due variabili d'ambiente:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """Sei un esperto barman e sommelier. Analizza questa foto e identifica TUTTE le bottiglie di alcolici, liquori, spirits, mixer e bevande che vedi.

Per OGNI bottiglia trovata rispondi con questo formato JSON esatto (e NIENTE altro testo):
{
  "detections": [
    {
      "name": "Nome Marca Esatto",
      "confidence": 0.95,
      "category": "Categoria"
    }
  ]
}

Le categorie possibili sono:
- Vodka
- Gin
- Rum
- Whisky
- Tequila
- Aperitivi e Liquori
- Vermouth e Amari
- Bollicine e Vino
- Mixer
- Birra
- Altro

Regole importanti:
- Identifica OGNI singola bottiglia visibile, anche se parzialmente nascosta
- Usa il nome commerciale esatto (es. "Jack Daniel's Tennessee Honey", non solo "whiskey")
- Se non riesci a leggere l'etichetta ma riconosci la forma/colore, prova comunque
- La confidence va da 0.0 a 1.0 (1.0 = sicurissimo)
- Rispondi SOLO con il JSON, nessun altro testo prima o dopo"""


def analyze_with_openai(image_base64: str) -> dict:
    """Analizza l'immagine usando GPT-4o"""
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.2
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return extract_json(text)


def analyze_with_anthropic(image_base64: str) -> dict:
    """Analizza l'immagine usando Claude"""
    headers = {
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    }
                ]
            }
        ]
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    return extract_json(text)


def extract_json(text: str) -> dict:
    """Estrae il JSON dalla risposta AI, anche se c'è testo extra"""
    # Prova prima il parsing diretto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Cerca un blocco JSON nella risposta
    match = re.search(r'\{[\s\S]*"detections"[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"detections": []}


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SELFY BAR Vision API"})


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Riceve: { "image_base64": "..." }
    Restituisce: { "detections": [ { "name", "confidence", "category" } ] }
    """
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "Manca image_base64 nel body"}), 400

        image_b64 = data["image_base64"]

        # Scegli il provider AI disponibile
        if ANTHROPIC_KEY:
            result = analyze_with_anthropic(image_b64)
        elif OPENAI_KEY:
            result = analyze_with_openai(image_b64)
        else:
            return jsonify({"error": "Nessuna API key configurata (OPENAI_API_KEY o ANTHROPIC_API_KEY)"}), 500

        # Valida e pulisci il risultato
        detections = []
        for item in result.get("detections", []):
            detections.append({
                "name": str(item.get("name", "Sconosciuto")),
                "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
                "category": str(item.get("category", "Altro"))
            })

        return jsonify({"detections": detections})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Timeout dal servizio AI. Riprova."}), 504
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Errore API AI: {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": f"Errore interno: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
