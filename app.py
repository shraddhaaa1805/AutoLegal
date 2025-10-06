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
import time

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(" Gemini API configured successfully")
else:
    logger.error(" GEMINI_API_KEY not found in environment variables")

app = Flask(__name__, static_folder='static')

# Enhanced CORS configuration
CORS(app,
     origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000", "http://127.0.0.1:5000"],
     supports_credentials=True,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

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

# Lazy initialization for performance
_reader_instance = None
_gemini_model_instance = None


def get_reader():
    global _reader_instance
    if _reader_instance is None:
        logger.info(" Initializing EasyOCR (first time only)...")
        try:
            _reader_instance = easyocr.Reader(['en'])
            logger.info(" EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f" EasyOCR initialization failed: {e}")
    return _reader_instance


# Use available Gemini models
AVAILABLE_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite-preview-06-17",
    "models/gemini-2.5-pro-preview-05-06"
]


def get_gemini_model():
    """Get the first available Gemini model"""
    for model_name in AVAILABLE_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a simple prompt
            test_response = model.generate_content("Say OK")
            if test_response.text:
                logger.info(f" Using model: {model_name}")
                return model
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
    logger.error(" No Gemini models available")
    return None


def get_gemini():
    global _gemini_model_instance
    if _gemini_model_instance is None:
        logger.info(" Initializing Gemini model (first time only)...")
        _gemini_model_instance = get_gemini_model()
    return _gemini_model_instance


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


# Health check endpoint with improved response
@app.route("/api/health", methods=["GET", "OPTIONS"])
def health_check():
    if request.method == "OPTIONS":
        return "", 200

    try:
        # Test if services are responsive
        gemini_status = "available" if get_gemini() else "unavailable"
        ocr_status = "available" if get_reader() else "unavailable"
        news_status = "available" if NEWS_API_KEY else "unavailable"

        return jsonify({
            "status": "running",
            "timestamp": time.time(),
            "services": {
                "gemini": gemini_status,
                "ocr": ocr_status,
                "news": news_status
            },
            "backend": "ready"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Handle preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


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


@app.route('/api/google-login', methods=['POST', 'OPTIONS'])
def google_login():
    if request.method == "OPTIONS":
        return "", 200

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


@app.route("/logout", methods=["POST", "OPTIONS"])
def logout():
    if request.method == "OPTIONS":
        return "", 200
    session.clear()
    logger.info("User logged out")
    return jsonify({"message": "Logged out"})


# Gemini AI helper with robust error handling
def ask_gemini(prompt):
    gemini_model = get_gemini()
    if not gemini_model:
        return "⚠ Gemini AI service is not available. Please check the configuration."

    try:
        logger.info(f"Gemini prompt: {prompt[:60]}...")
        response = gemini_model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return " Gemini returned no response."
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return f"️ Gemini Error: {str(e)}"


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided"}), 400

    enhanced_prompt = f"""You are AutoLegal AI, a helpful legal assistant for Indian law. 
    Provide accurate, helpful guidance while being clear about limitations.

    User Question: {query}

    Please provide a comprehensive but concise answer. If the question involves specific legal advice, 
    explain general principles and suggest consulting a qualified lawyer.

    Always include this disclaimer at the end:
    "Disclaimer: This is informational guidance only and not formal legal advice. For official legal matters, please consult a qualified lawyer."

    Answer:"""

    response_text = ask_gemini(enhanced_prompt)
    return jsonify({"response": response_text})


@app.route("/api/simplify", methods=["POST", "OPTIONS"])
def simplify():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"simplified": "No text provided"}), 400

    enhanced_prompt = f"""Simplify this legal clause into plain English that a non-lawyer can understand:

    Legal Clause: {text}

    Provide a clear, simple explanation:"""

    response_text = ask_gemini(enhanced_prompt)
    return jsonify({"simplified": response_text})


@app.route("/api/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"summary": "No text provided"}), 400

    enhanced_prompt = f"""Summarize this contract in clear bullet points:

    Contract Text: {text}

    Provide a concise summary with key points:"""

    response_text = ask_gemini(enhanced_prompt)
    return jsonify({"summary": response_text})


@app.route("/api/risk", methods=["POST", "OPTIONS"])
def risk():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"risks": "No text provided"}), 400

    enhanced_prompt = f"""Identify potential risks and issues in this legal text:

    Text: {text}

    List the main risks and concerns:"""

    response_text = ask_gemini(enhanced_prompt)
    return jsonify({"risks": response_text})


@app.route("/api/compliance", methods=["POST", "OPTIONS"])
def compliance():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"compliance": "No text provided"}), 400

    enhanced_prompt = f"""Check this document for compliance issues under Indian law:

    Document Text: {text}

    Identify any compliance issues:"""

    response_text = ask_gemini(enhanced_prompt)
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


@app.route("/api/ocr", methods=["POST", "OPTIONS"])
def ocr():
    if request.method == "OPTIONS":
        return "", 200

    base64_image = request.json.get("imageData", "")
    if not base64_image:
        logger.warning("OCR called without image data")
        return jsonify({"error": "No image data provided"}), 400

    img = read_image_from_base64(base64_image)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        reader = get_reader()
        if reader is None:
            return jsonify({"error": "OCR service not available"}), 500

        results = reader.readtext(img)
        text = " ".join([res[1] for res in results])
        logger.info(f"OCR extracted text length: {len(text)}")
        return jsonify({"text": text.strip()})
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return jsonify({"error": "OCR processing failed", "message": str(e)}), 500


# Legal news endpoint
@app.route("/api/news", methods=["POST", "OPTIONS"])
def legal_updates():
    if request.method == "OPTIONS":
        return "", 200

    topic = request.json.get("topic", "indian law")
    if not NEWS_API_KEY:
        return jsonify({"error": "News API not configured"}), 500

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
    print(" Starting AutoLegal Backend Server...")
    print(f" Server: http://127.0.0.1:5000")
    print(f" CORS enabled for: http://localhost:3000, http://127.0.0.1:3000")

    # Test services
    print(f" Gemini: {'Available' if get_gemini() else ' Unavailable'}")
    print(f" News API: {' Available' if NEWS_API_KEY else ' Unavailable'}")
    print(f" OCR: {' Available' if get_reader() else ' Unavailable'}")

    app.run(debug=True, host="127.0.0.1", port=5000)
