import logging
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import google.generativeai as genai
import easyocr
import numpy as np
import cv2
import base64
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from google_auth_oauthlib.flow import Flow
import json

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')

# Google OAuth config
CLIENT_SECRETS_FILE = os.path.join(os.path.dirname(__file__), "client_secret_999.json")

try:
    with open(CLIENT_SECRETS_FILE) as f:
        client_secrets_data = json.load(f)
    CLIENT_ID = client_secrets_data['web']['client_id']
except Exception as e:
    logger.error(f"Error loading client secrets json: {e}")
    CLIENT_ID = None

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

# Initialize EasyOCR once for performance
reader = easyocr.Reader(['en'])


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


# Google OAuth login flow
@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("callback", _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session["state"] = state
    logger.info("Redirecting to Google OAuth URL")
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=session.get("state"),
        redirect_uri=url_for("callback", _external=True)
    )
    try:
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        session["credentials"] = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes
        }
        logger.info("Google OAuth login successful")
        return jsonify({"message": "Login successful", "credentials": session["credentials"]})
    except Exception as e:
        logger.error(f"Error in Google OAuth callback: {e}")
        return jsonify({"error": "OAuth callback failed", "message": str(e)}), 400


@app.route('/api/google-login', methods=['POST'])
def google_login():
    token = request.json.get('credential')
    if CLIENT_ID is None:
        logger.error("CLIENT_ID not configured")
        return jsonify({'success': False, 'message': 'Server misconfiguration'}), 500
    try:
        id_info = id_token.verify_oauth2_token(token, grequests.Request(), CLIENT_ID)
        session['user'] = {
            'id': id_info['sub'],
            'email': id_info.get('email'),
            'name': id_info.get('name', '')
        }
        logger.info(f"Google login success for user {session['user']['email']}")
        return jsonify({'success': True, 'user': session['user']})
    except ValueError as e:
        logger.warning(f"Invalid Google token: {e}")
        return jsonify({'success': False, 'message': 'Invalid token', 'error': str(e)}), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    logger.info("User logged out")
    return jsonify({"message": "Logged out"})


# Gemini AI helper with robust error handling
def ask_gemini(prompt):
    try:
        logger.info(f"Gemini prompt: {prompt[:60]}...")  # log short preview
        model = genai.GenerativeModel("gemini-2.0-flash")  # use stable available model
        response = model.generate_content(prompt)

        # Safely get text
        if hasattr(response, "text") and response.text:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "⚠️ Gemini returned no response."
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return "⚠️ Gemini is temporarily unavailable. Please try again later."


@app.route("/api/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided"}), 400
    response_text = ask_gemini(f"You are AutoLegal AI assistant. Answer with disclaimer (not legal advice):\n{query}")
    return jsonify({"response": response_text})


@app.route("/api/simplify", methods=["POST"])
def simplify():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"simplified": "No text provided"}), 400
    response_text = ask_gemini(f"Simplify this legal clause for a layman:\n{text}")
    return jsonify({"simplified": response_text})


@app.route("/api/summarize", methods=["POST"])
def summarize():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"summary": "No text provided"}), 400
    response_text = ask_gemini(f"Summarize this contract in clear short points:\n{text}")
    return jsonify({"summary": response_text})


@app.route("/api/risk", methods=["POST"])
def risk():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"risks": "No text provided"}), 400
    response_text = ask_gemini(f"Identify risks and potential issues in this text:\n{text}")
    return jsonify({"risks": response_text})


@app.route("/api/compliance", methods=["POST"])
def compliance():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"compliance": "No text provided"}), 400
    response_text = ask_gemini(f"Check compliance issues in this document under Indian law:\n{text}")
    return jsonify({"compliance": response_text})


# OCR Endpoint
def read_image_from_base64(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


@app.route("/api/ocr", methods=["POST"])
def ocr():
    base64_image = request.json.get("imageData", "")
    if not base64_image:
        logger.warning("OCR called without image data")
        return jsonify({"error": "No image data provided"}), 400

    img = read_image_from_base64(base64_image)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        results = reader.readtext(img)
        text = " ".join([res[1] for res in results])
        logger.info(f"OCR extracted text length: {len(text)}")
        return jsonify({"text": text.strip()})
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return jsonify({"error": "OCR processing failed", "message": str(e)}), 500


# Legal news endpoint
@app.route("/api/news", methods=["POST"])
def legal_updates():
    topic = request.json.get("topic", "indian law")
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            articles = res.json().get("articles", [])
            top_updates = [f"{a['title']} - {a['source']['name']}" for a in articles[:5]]
            logger.info(f"News API returned {len(top_updates)} items for topic: {topic}")
            return jsonify({"updates": top_updates})
        else:
            logger.error(f"News API failed with status {res.status_code}")
            return jsonify({"error": "News API failed", "code": res.status_code}), 502
    except Exception as e:
        logger.error(f"News API request error: {e}")
        return jsonify({"error": "News API error", "message": str(e)}), 502


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
