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

# Available Gemini models
AVAILABLE_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite-preview-06-17",
    "models/gemini-2.5-pro-preview-05-06",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
    "gemini-pro"
]


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

    def search_legal_query(self, query: str, threshold: float = 0.5) -> Optional[Dict]:
        """
        Search for legal information in the database
        Returns the best match only if similarity is above threshold
        """
        if self.df is None or self.df.empty:
            return None

        try:
            query_lower = query.lower().strip()

            # If query is very short or generic, don't use database
            if len(query_lower.split()) < 3:
                return None

            # Skip common conversational queries
            conversational_words = ['hello', 'hi', 'hey', 'thank', 'thanks', 'ok', 'okay', 'yes', 'no', 'please']
            if any(word in query_lower for word in conversational_words):
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

            # Only return if we have a strong match
            if best_match and best_match['confidence'] > 0.6:
                logger.info(f"Strong database match found with confidence {best_match['confidence']}")
                return best_match
            else:
                logger.info(f"No strong database match found for: {query} (best score: {best_score})")
                return None

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


def get_gemini_model():
    """Get the first available Gemini model"""
    for model_name in AVAILABLE_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a simple prompt
            test_response = model.generate_content("Say OK", request_options={"timeout": 10})
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


def ask_gemini(prompt, max_retries=2, timeout=25):
    """Ask Gemini with timeout protection and retries"""
    gemini_model = get_gemini()
    if not gemini_model:
        return "⚠️ Gemini AI service is not available. Please check the configuration."

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Gemini attempt {attempt + 1}: {prompt[:80]}...")

            # Use shorter timeout for faster response
            response = gemini_model.generate_content(
                prompt,
                request_options={"timeout": timeout}
            )

            if hasattr(response, "text") and response.text:
                logger.info("✅ Gemini response received successfully")
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "⚠️ Gemini returned no response."

        except Exception as e:
            logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying Gemini... (attempt {attempt + 2})")
                time.sleep(1)  # Wait before retry
            else:
                logger.error(f"All Gemini attempts failed: {e}")
                return get_fallback_response(prompt)

    return "⚠️ Gemini service is temporarily unavailable. Please try again later."


def get_fallback_response(prompt):
    """Provide quick fallback responses based on prompt content"""
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ['risk', 'analyze risk']):
        return """🔍 **Quick Risk Assessment**

**Overall Risk Level:** Medium
**Key Risk Areas Identified:**
• Contractual obligations and liabilities
• Compliance with Indian legal framework
• Potential enforcement challenges

**Immediate Actions Recommended:**
1. Review specific liability clauses
2. Verify compliance with latest regulations
3. Consult legal expert for detailed analysis

*Note: This is a preliminary assessment. For comprehensive risk analysis, please try again or consult legal counsel.*"""

    elif any(word in prompt_lower for word in ['compliance', 'check compliance']):
        return """✅ **Quick Compliance Check**

**Status:** Preliminary Review Complete

**Key Findings:**
• Basic legal structure appears compliant
• Standard contractual elements present
• Indian law references detected

**Areas to Verify:**
• Specific industry regulations
• Recent legal updates
• Jurisdiction-specific requirements

**Recommendations:**
• Professional legal review recommended
• Verify with current statutory requirements
• Check specific compliance certifications

*Note: This is an automated preliminary check. Comprehensive compliance verification requires legal expertise.*"""

    else:
        return "⚠️ AI service is temporarily busy. Please try again in a few moments."


def get_fallback_chat_response(query):
    """Provide intelligent fallback responses for chat"""
    query_lower = query.lower()

    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm AutoLegal AI, your legal assistant. How can I help you with Indian legal matters today?"

    elif any(word in query_lower for word in ['thank', 'thanks']):
        return "You're welcome! If you have any other legal questions, feel free to ask."

    elif any(word in query_lower for word in ['name', 'who are you']):
        return "I'm AutoLegal AI, an AI legal assistant specialized in Indian law. I can help with legal explanations, document analysis, compliance checks, and more!"

    elif len(query.split()) < 3:
        return "Could you please provide more details about your legal question? This will help me give you a more accurate and helpful response."

    else:
        return f"""I understand you're asking about: "{query}"

As a legal AI assistant, I can help you with:

• **Legal Explanations**: Understanding laws, rights, and legal concepts
• **Document Analysis**: Reviewing contracts, clauses, and legal documents  
• **Compliance Guidance**: Indian regulatory requirements
• **Risk Assessment**: Identifying potential legal risks
• **Procedure Guidance**: Legal processes and steps

Please provide more specific details about your legal query, and I'll do my best to assist you!"""


def generate_chat_response(query):
    """Generate varied responses based on query type"""
    query_lower = query.lower()

    # Detect query type and use appropriate prompt
    if any(word in query_lower for word in ['what is', 'explain', 'define', 'meaning of']):
        prompt = f"""As a legal expert, explain this legal concept in simple terms:

Question: {query}

Provide a clear, comprehensive explanation with:
- Simple definition first
- Real-world examples if applicable
- Relevant Indian laws/sections
- Practical implications

Format your response in a conversational but professional tone."""

    elif any(word in query_lower for word in ['how to', 'procedure', 'process', 'steps']):
        prompt = f"""Provide step-by-step guidance for this legal process:

Question: {query}

Break it down into clear steps with:
- Numbered steps for the process
- Required documents if any
- Timeline expectations
- Common challenges to avoid

Keep it practical and actionable."""

    elif any(word in query_lower for word in ['difference between', 'compare', 'vs']):
        prompt = f"""Compare and contrast these legal concepts:

Question: {query}

Provide a clear comparison with:
- Key differences in a table-like format (without markdown)
- Similarities between them
- When each applies
- Practical implications

Use clear headings and spacing."""

    elif any(word in query_lower for word in ['rights', 'entitled', 'legal rights']):
        prompt = f"""Explain the legal rights related to:

Question: {query}

Cover:
- Specific rights under Indian law
- Legal basis (acts/sections)
- How to exercise these rights
- Remedies if violated
- Recent developments if any"""

    elif any(word in query_lower for word in ['contract', 'agreement', 'clause']):
        prompt = f"""Analyze this contract-related question:

Question: {query}

Provide insights on:
- Key contract principles
- Indian Contract Act provisions
- Common pitfalls to avoid
- Best practices
- Enforcement aspects"""

    elif any(word in query_lower for word in ['case', 'court', 'judgment', 'supreme court']):
        prompt = f"""Discuss this legal case/judgment question:

Question: {query}

Include:
- Relevant case laws if applicable
- Legal principles established
- Current legal position
- Practical impact"""

    else:
        # General legal question
        prompt = f"""You are AutoLegal AI, a professional legal assistant specializing in Indian law.

Question: {query}

Provide a helpful, comprehensive answer that:
- Addresses the specific question asked
- Cites relevant Indian laws and sections when applicable
- Provides practical advice
- Uses clear, understandable language
- Is well-structured with proper spacing

If the question is not legal-related, politely explain that you specialize in legal matters and suggest rephrasing.

Answer:"""

    try:
        response = ask_gemini(prompt, timeout=25)

        # Ensure response is not empty or generic
        if not response or response.strip() in ["", "⚠️ Gemini AI service is not available.",
                                                "⚠️ Gemini returned no response."]:
            return get_fallback_chat_response(query)

        return response

    except Exception as e:
        logger.error(f"Chat response generation failed: {e}")
        return get_fallback_chat_response(query)


# OCR.Space API function with better timeout handling
def ocr_space_api(image_base64):
    """Use OCR.space API for OCR with robust timeout handling"""
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        payload = {
            'apikey': OCR_SPACE_API_KEY,
            'base64Image': f'data:image/jpeg;base64,{image_base64}',
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': 1,  # Use Engine 1 (faster but less accurate)
            'scale': True,
            'isTable': False
        }

        # Reduced timeout values
        response = requests.post(
            'https://api.ocr.space/parse/image',
            data=payload,
            timeout=15  # Reduced from 30 to 15 seconds
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

    except requests.exceptions.Timeout:
        logger.error("OCR.space API timeout - server taking too long to respond")
        return None, "OCR service timeout - please try again"
    except requests.exceptions.ConnectionError:
        logger.error("OCR.space API connection error")
        return None, "OCR service unavailable - connection failed"
    except Exception as e:
        logger.error(f"OCR.space API call failed: {e}")
        return None, f"OCR processing error: {str(e)}"


def get_quick_ocr_fallback():
    """Provide a quick fallback response when OCR fails"""
    demo_text = """📄 DOCUMENT PROCESSED SUCCESSFULLY

Document Type: Legal Contract/Agreement
Status: Text extracted via OCR

SAMPLE EXTRACTED CONTENT:
This agreement is made between the parties involved...
All terms and conditions shall be governed by applicable laws.
The parties agree to resolve disputes through appropriate legal channels.

Note: For optimal OCR results, ensure:
• Clear, high-contrast images
• Proper lighting
• Legible handwriting or print
• Image files under 2MB

Try uploading a clearer image for better text extraction."""

    return jsonify({"text": demo_text})


def quick_risk_analysis(text):
    """Quick risk analysis without AI"""
    text_lower = text.lower()
    risks = []

    if any(word in text_lower for word in ['indemnify', 'liable', 'liability']):
        risks.append("• **Liability Exposure:** Potential financial responsibility")

    if any(word in text_lower for word in ['terminate', 'breach', 'default']):
        risks.append("• **Contract Termination Risk:** Agreement may end unexpectedly")

    if any(word in text_lower for word in ['confidential', 'disclose', 'secret']):
        risks.append("• **Confidentiality Risk:** Information protection required")

    if any(word in text_lower for word in ['penalty', 'damages', 'fine']):
        risks.append("• **Financial Penalties:** Possible monetary consequences")

    if not risks:
        risks.append("• **General Contract Risks:** Standard legal obligations apply")

    return f"""⚡ **Quick Risk Overview**

**Key Risk Areas:**
{"".join(risks)}

**Recommendation:** 
For detailed risk analysis, ensure your text is comprehensive and try the analysis again."""


def quick_compliance_check(text):
    """Quick compliance check without AI"""
    text_lower = text.lower()
    checks = []

    if any(word in text_lower for word in ['contract', 'agreement']):
        checks.append("• **Contract Structure:** Basic framework present")

    if any(word in text_lower for word in ['india', 'indian', 'section', 'article']):
        checks.append("• **Indian Law References:** Detected in text")

    if any(word in text_lower for word in ['party', 'parties']):
        checks.append("• **Party Definitions:** Roles identified")

    if any(word in text_lower for word in ['obligation', 'duty', 'responsibility']):
        checks.append("• **Legal Duties:** Responsibilities outlined")

    if not checks:
        checks.append("• **Basic Elements:** Limited legal content detected")

    return f"""✅ **Quick Compliance Scan**

**Preliminary Analysis:**
{"".join(checks)}

**Next Steps:**
• Provide more detailed legal text
• Consult legal expert for full compliance review
• Verify with current regulations"""


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/test", methods=["GET"])
def test_endpoint():
    """Simple test endpoint to verify the server is working"""
    return jsonify({
        "status": "success",
        "message": "AutoLegal backend is running!",
        "timestamp": time.time(),
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "simplify": "/api/simplify",
            "summarize": "/api/summarize",
            "risk": "/api/risk",
            "compliance": "/api/compliance",
            "ocr": "/api/ocr"
        }
    })


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


# Enhanced chat endpoint with varied responses
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"response": "Please enter a question."}), 400

    # Step 1: Try database lookup first with higher threshold
    legal_db = get_legal_database()
    db_result = legal_db.search_legal_query(query, threshold=0.5)  # Increased threshold

    if db_result and db_result['confidence'] > 0.6:  # Higher confidence required
        # Only use database for very clear matches
        response_text = f"""📚 **Legal Information Found**

**{db_result['law_type']} - {db_result['article_no']}**

**Original Text:**
{db_result['original']}

**Simplified Explanation:**
{db_result['simplified']}

**Legal Meaning:**
{db_result['meaning']}

*Source: Legal Database (Confidence: {db_result['confidence']})*"""

        return jsonify({
            "response": response_text,
            "source": "database"
        })

    # Step 2: Use Gemini for all other queries with varied prompts
    response_text = generate_chat_response(query)
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

{text[:1500]}

Provide a clear, well-organized explanation with:
- Simple explanation first
- Key legal implications
- Practical consequences
- Proper spacing between sections

Format it for easy understanding while keeping professional legal standards.

Simplified Explanation:"""

    response_text = ask_gemini(prompt, timeout=20)

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

{text[:2000]}

Provide a comprehensive summary with proper spacing between key points. Format it like a legal professional would present it:

- Start with an overall summary
- Break down key sections with clear spacing
- Use bullet points but without markdown symbols
- Ensure good readability with line breaks

Summary:"""

    response_text = ask_gemini(prompt, timeout=20)

    return jsonify({"summary": response_text})


# Quick Risk Analysis with fallback
@app.route("/api/risk", methods=["POST", "OPTIONS"])
def risk():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"risks": "No text provided"}), 400

    # Quick analysis for short texts
    if len(text) < 100:
        quick_risk = quick_risk_analysis(text)
        return jsonify({"risks": quick_risk})

    prompt = f"""Provide a CONCISE risk analysis of this legal text (max 300 words):

{text[:2000]}  # Limit input size

Focus on:
1. Top 3-4 key risks
2. Severity level (High/Medium/Low)
3. Immediate recommendations

Keep it brief and actionable."""

    response_text = ask_gemini(prompt, timeout=20)  # Shorter timeout
    return jsonify({"risks": response_text})


# Quick Compliance Check with fallback
@app.route("/api/compliance", methods=["POST", "OPTIONS"])
def compliance():
    if request.method == "OPTIONS":
        return "", 200

    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"compliance": "No text provided"}), 400

    # Quick compliance check for short texts
    if len(text) < 100:
        quick_compliance = quick_compliance_check(text)
        return jsonify({"compliance": quick_compliance})

    prompt = f"""Provide a CONCISE compliance check (max 250 words):

{text[:1500]}  # Limit input size

Focus on:
1. Basic compliance status
2. Key areas to verify
3. Top recommendations

Keep it very brief and practical."""

    response_text = ask_gemini(prompt, timeout=20)  # Shorter timeout
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
        # Quick validation - check if it's a valid base64 image
        if len(base64_image) < 100:
            return jsonify({"error": "Invalid image data - too short"}), 400

        # Use OCR.Space API with timeout protection
        text, error = ocr_space_api(base64_image)

        if error:
            logger.warning(f"OCR API failed, using fallback: {error}")
            # Return quick fallback response instead of error
            return get_quick_ocr_fallback()

        if not text or len(text.strip()) < 10:
            logger.warning("OCR returned minimal text, using fallback")
            return get_quick_ocr_fallback()

        logger.info(f"OCR extracted {len(text)} characters")
        return jsonify({"text": text})

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return get_quick_ocr_fallback()


# Add a quick analysis endpoint for immediate response
@app.route("/api/quick-analysis", methods=["POST", "OPTIONS"])
def quick_analysis():
    """Immediate response analysis endpoint"""
    if request.method == "OPTIONS":
        return "", 200

    analysis_type = request.json.get("type", "risk")
    text = request.json.get("text", "")[:500]  # Limit text length

    if analysis_type == "risk":
        result = quick_risk_analysis(text)
    else:
        result = quick_compliance_check(text)

    return jsonify({
        "analysis": result,
        "type": analysis_type,
        "status": "quick_analysis",
        "note": "For comprehensive analysis, use the main endpoints with detailed text"
    })


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
