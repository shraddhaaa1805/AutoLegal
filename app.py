import logging
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import google.generativeai as genai
import numpy as np
import cv2
import base64
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from google_auth_oauthlib.flow import Flow
import json
import time
import pandas as pd
import re
import csv
from typing import Dict, Optional

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "K87899142388957")  # Free public key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
else:
    logger.error("GEMINI_API_KEY not found in environment variables")

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
_gemini_model_instance = None
_legal_db_instance = None


class LegalDatabase:
    def __init__(self, csv_path: str = "legal_data/laws.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_database()

    def load_database(self):
        """Load the legal database from CSV with robust error handling"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"Legal database file not found: {self.csv_path}")
                self.df = pd.DataFrame()
                return

            # Try multiple methods to read the CSV
            success = False

            # Method 1: Try standard pandas read
            try:
                self.df = pd.read_csv(self.csv_path)
                success = True
                logger.info("CSV loaded with standard pandas read")
            except Exception as e:
                logger.warning(f"Standard read failed: {e}")

            # Method 2: Try with error handling for bad lines
            if not success:
                try:
                    self.df = pd.read_csv(self.csv_path, on_bad_lines='skip', encoding='utf-8')
                    success = True
                    logger.info("CSV loaded with error handling")
                except Exception as e:
                    logger.warning(f"Error-handling read failed: {e}")

            # Method 3: Manual CSV reading as last resort
            if not success:
                try:
                    data = []
                    with open(self.csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        headers = next(reader)  # Get header row

                        for line_num, row in enumerate(reader, 2):
                            if len(row) == 5:  # Only take valid rows with exactly 5 columns
                                data.append(row)
                            else:
                                logger.warning(f"Skipping invalid row {line_num}: has {len(row)} columns instead of 5")

                    if data:
                        self.df = pd.DataFrame(data, columns=headers)
                        success = True
                        logger.info("CSV loaded with manual reading")
                    else:
                        logger.error("No valid data found in CSV")
                except Exception as e:
                    logger.error(f"Manual CSV reading failed: {e}")

            if success and self.df is not None and not self.df.empty:
                # Clean the data
                self.df = self.df.fillna('')
                self.df = self.df.drop_duplicates()

                # Ensure we have the expected columns
                expected_columns = ['law_type', 'original', 'simplified', 'article_no', 'meaning']
                for col in expected_columns:
                    if col not in self.df.columns:
                        self.df[col] = ''

                logger.info(f"Legal database loaded with {len(self.df)} entries")
                logger.info(f"Columns found: {list(self.df.columns)}")
            else:
                logger.error("Failed to load any data from CSV")
                self.df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Critical error loading legal database: {e}")
            self.df = pd.DataFrame()

    def search_legal_query(self, query: str, threshold: float = 0.3) -> Optional[Dict]:
        """
        Search for legal information in the database
        Returns the best match if similarity is above threshold
        """
        if self.df is None or self.df.empty:
            return None

        try:
            query_lower = query.lower().strip()

            # If query is very short, use API directly
            if len(query_lower.split()) < 2:
                return None

            best_match = None
            best_score = 0

            for _, row in self.df.iterrows():
                score = self._calculate_similarity(query_lower, row)
                if score > best_score and score > threshold:
                    best_score = score
                    best_match = {
                        'law_type': str(row.get('law_type', '')),
                        'original': str(row.get('original', '')),
                        'simplified': str(row.get('simplified', '')),
                        'article_no': str(row.get('article_no', '')),
                        'meaning': str(row.get('meaning', '')),
                        'confidence': round(best_score, 2),
                        'source': 'database'
                    }

            if best_match:
                logger.info(
                    f"Database match found: {best_match['law_type']} - {best_match['original'][:50]}... with confidence {best_match['confidence']}")
            else:
                logger.info(f"No database match found for: {query}")

            return best_match

        except Exception as e:
            logger.error(f"Error searching legal database: {e}")
            return None

    def _calculate_similarity(self, query: str, row) -> float:
        """Calculate similarity score between query and database entry"""
        try:
            # Combine relevant text from all columns for matching
            relevant_text = ""
            for col in ['law_type', 'original', 'simplified', 'article_no', 'meaning']:
                if col in row and row[col] and str(row[col]).strip():
                    relevant_text += " " + str(row[col]).lower()

            # Simple word overlap scoring
            query_words = set(re.findall(r'\w+', query))
            text_words = set(re.findall(r'\w+', relevant_text))

            if not query_words:
                return 0

            overlap = len(query_words.intersection(text_words))
            base_score = overlap / len(query_words)

            # Boost score if query contains law section numbers
            if re.search(r'\d+', query) and str(row.get('article_no', '')) in query:
                base_score += 0.3

            return min(base_score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0

    def get_law_by_section(self, section: str) -> Optional[Dict]:
        """Get law information by section number"""
        if self.df is None or self.df.empty:
            return None

        try:
            # Clean section input
            section_clean = str(section).strip()
            # Try exact match first, then partial match
            for _, row in self.df.iterrows():
                article_no = str(row.get('article_no', ''))
                if section_clean == article_no or section_clean in article_no or article_no in section_clean:
                    return {
                        'law_type': str(row.get('law_type', '')),
                        'original': str(row.get('original', '')),
                        'simplified': str(row.get('simplified', '')),
                        'article_no': str(row.get('article_no', '')),
                        'meaning': str(row.get('meaning', '')),
                        'source': 'database'
                    }
            return None
        except Exception as e:
            logger.error(f"Error searching law by section: {e}")
            return None


def get_legal_database():
    global _legal_db_instance
    if _legal_db_instance is None:
        logger.info("Initializing Legal Database (first time only)...")
        _legal_db_instance = LegalDatabase()
    return _legal_db_instance


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
                logger.info(f"Using model: {model_name}")
                return model
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
    logger.error("No Gemini models available")
    return None


def get_gemini():
    global _gemini_model_instance
    if _gemini_model_instance is None:
        logger.info("Initializing Gemini model (first time only)...")
        _gemini_model_instance = get_gemini_model()
    return _gemini_model_instance


# OCR.Space API function
def ocr_space_api(image_base64):
    """Use OCR.space API for OCR without local installation"""
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        payload = {
            'apikey': OCR_SPACE_API_KEY,
            'base64Image': f'data:image/jpeg;base64,{image_base64}',
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 2  # Engine 2 is more accurate
        }

        response = requests.post(
            'https://api.ocr.space/parse/image',
            data=payload,
            timeout=30
        )

        result = response.json()

        if result['IsErroredOnProcessing']:
            error_message = result.get('ErrorMessage', 'Unknown OCR error')
            logger.error(f"OCR.space API error: {error_message}")
            return None, error_message

        if 'ParsedResults' not in result or not result['ParsedResults']:
            return None, "No OCR results returned"

        text = result['ParsedResults'][0]['ParsedText']
        return text.strip(), None

    except Exception as e:
        logger.error(f"OCR.space API call failed: {e}")
        return None, str(e)


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
        legal_db = get_legal_database()
        db_status = "available" if legal_db.df is not None and not legal_db.df.empty else "unavailable"
        db_entries = len(legal_db.df) if legal_db.df is not None else 0

        gemini_status = "available" if get_gemini() else "unavailable"
        ocr_status = "available"  # OCR.Space API is always available
        news_status = "available" if NEWS_API_KEY else "unavailable"

        return jsonify({
            "status": "running",
            "timestamp": time.time(),
            "services": {
                "gemini": gemini_status,
                "ocr": ocr_status,
                "news": news_status,
                "legal_database": db_status
            },
            "database_entries": db_entries,
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


# Enhanced chat endpoint with database-first approach
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    query = request.json.get("query", "")
    if not query:
        return jsonify({"response": "No query provided"}), 400

    # Step 1: Try database lookup first
    legal_db = get_legal_database()
    db_result = legal_db.search_legal_query(query)

    if db_result and db_result['confidence'] > 0.4:
        response_text = f"""{db_result['simplified']}

Meaning: {db_result['meaning']}

Law Type: {db_result['law_type']}
Article/Section: {db_result['article_no']}"""

        return jsonify({
            "response": response_text,
            "source": "database"
        })

    # Step 2: Fallback to Gemini API with professional formatting
    enhanced_prompt = f"""You are AutoLegal AI, a professional legal assistant for Indian law.

Question: {query}

Provide a comprehensive, well-structured answer with:
- Clear explanation of legal principles
- Relevant Indian laws and sections if applicable
- Practical implications
- Proper spacing between different aspects
- Professional legal language

Format your response like a legal expert advising a client, with good organization and readability.

Answer:"""

    response_text = ask_gemini(enhanced_prompt)

    return jsonify({
        "response": response_text,
        "source": "ai"
    })


# Enhanced simplify endpoint with database lookup
@app.route("/api/simplify", methods=["POST", "OPTIONS"])
def simplify():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"simplified": "No text provided"}), 400

    legal_db = get_legal_database()

    # Try database first
    db_result = legal_db.search_legal_query(text)
    if db_result and db_result['confidence'] > 0.6:
        simplified_text = f"""Simplified Explanation:
{db_result['simplified']}

Detailed Meaning:
{db_result['meaning']}

Law Type: {db_result['law_type']}
Relevant Section: {db_result['article_no']}"""

        return jsonify({
            "simplified": simplified_text
        })

    # Check for law sections
    section_patterns = [r'section\s+(\d+[A-Z]*)', r'article\s+(\d+[A-Z]*)']
    for pattern in section_patterns:
        section_match = re.search(pattern, text, re.IGNORECASE)
        if section_match:
            section = section_match.group(1)
            db_result = legal_db.get_law_by_section(section)
            if db_result:
                simplified_text = f"""Legal Section {db_result['article_no']}:
{db_result['simplified']}

Meaning:
{db_result['meaning']}

Law Type: {db_result['law_type']}"""

                return jsonify({
                    "simplified": simplified_text
                })

    # Fallback to Gemini with professional formatting
    prompt = f"""Simplify this legal clause into plain English while maintaining professional structure:

{text}

Provide a clear, well-organized explanation with:
- Simple explanation first
- Key legal implications
- Practical consequences
- Proper spacing between sections

Format it for easy understanding while keeping professional legal standards.

Simplified Explanation:"""

    response_text = ask_gemini(prompt)

    return jsonify({
        "simplified": response_text
    })


# Professional Summarize Endpoint
@app.route("/api/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"summary": "No text provided"}), 400

    prompt = f"""Summarize this legal document in a clear, structured way:

{text}

Provide a comprehensive summary with proper spacing between key points. Format it like a legal professional would present it:

- Start with an overall summary
- Break down key sections with clear spacing
- Use bullet points but without markdown symbols
- Ensure good readability with line breaks

Summary:"""

    response_text = ask_gemini(prompt)

    return jsonify({"summary": response_text})


# Professional Risk Analysis Endpoint
@app.route("/api/risk", methods=["POST", "OPTIONS"])
def risk():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"risks": "No text provided"}), 400

    prompt = f"""Analyze potential risks in this legal text and present them in a structured, professional format:

{text}

Provide a comprehensive risk analysis with:

1. Overall risk assessment
2. Specific risks categorized (high/medium/low priority)
3. Clear explanations for each risk
4. Proper spacing between different risk categories
5. Practical recommendations

Present it in a way that a lawyer would to a client, with clear organization and readability.

Risk Analysis:"""

    response_text = ask_gemini(prompt)

    return jsonify({"risks": response_text})


# Professional Compliance Checker Endpoint
@app.route("/api/compliance", methods=["POST", "OPTIONS"])
def compliance():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "")
    if not text:
        return jsonify({"compliance": "No text provided"}), 400

    prompt = f"""Check this document for compliance with Indian laws and present findings professionally:

{text}

Provide a comprehensive compliance review with:

- Overall compliance status
- Specific compliance issues found
- Relevant Indian laws and sections
- Recommendations for compliance
- Potential legal consequences

Format the response with clear sections, proper spacing, and professional legal language. Make it suitable for presenting to legal counsel.

Compliance Review:"""

    response_text = ask_gemini(prompt)

    return jsonify({"compliance": response_text})


# OCR Endpoint with OCR.Space API
@app.route("/api/ocr", methods=["POST", "OPTIONS"])
def ocr():
    if request.method == "OPTIONS":
        return "", 200

    base64_image = request.json.get("imageData", "")
    if not base64_image:
        logger.warning("OCR called without image data")
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Use OCR.Space API
        text, error = ocr_space_api(base64_image)

        if error:
            logger.error(f"OCR API failed: {error}")
            return jsonify({"error": f"OCR processing failed: {error}"}), 500

        if not text:
            return jsonify({"error": "No text detected in the image. Please try a clearer image."}), 400

        logger.info(f"OCR extracted {len(text)} characters")
        return jsonify({"text": text})

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return jsonify({"error": f"OCR processing failed: {str(e)}"}), 500


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
            updates = [f"{a['title']}" for a in articles[:5]]
            logger.info(f"News API returned {len(updates)} items for topic: {topic}")
            return jsonify({"updates": updates})
        else:
            logger.error(f"News API failed with status {res.status_code}")
            return jsonify({"error": "News API failed", "code": res.status_code}), 502
    except Exception as e:
        logger.error(f"News API request error: {e}")
        return jsonify({"error": "News API error", "message": str(e)}), 502


if __name__ == "__main__":
    print("Starting AutoLegal Backend Server...")
    print(f"Server: http://127.0.0.1:5000")
    print(f"CORS enabled for: http://localhost:3000, http://127.0.0.1:3000")

    # Pre-initialize services
    print("Pre-initializing services...")

    # Initialize services
    legal_db = get_legal_database()
    gemini = get_gemini()

    print(f"OCR: Available (OCR.Space API)")
    print(f"Gemini: {'Available' if gemini else 'Unavailable'}")
    print(f"News API: {'Available' if NEWS_API_KEY else 'Unavailable'}")
    print(f"Legal Database: {'Available' if legal_db.df is not None and not legal_db.df.empty else 'Unavailable'}")
    if legal_db.df is not None:
        print(f"Database Entries: {len(legal_db.df)}")

    app.run(debug=True, host="127.0.0.1", port=5000)
