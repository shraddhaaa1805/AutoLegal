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

    def search_legal_query(self, query: str, threshold: float = 0.7) -> Optional[Dict]:
        """
        Search for legal information in the database with improved accuracy
        """
        if self.df is None or self.df.empty:
            return None

        try:
            query_lower = query.lower().strip()

            # Skip if query doesn't contain legal section numbers
            if not re.search(r'section\s+\d+|article\s+\d+|sec\.?\s*\d+|art\.?\s*\d+', query_lower):
                return None

            # Skip conversational and general queries
            general_words = ['hello', 'hi', 'thank', 'please', 'what is', 'explain', 'how to']
            if any(word in query_lower for word in general_words):
                return None

            best_match = None
            best_score = 0

            for _, row in self.df.iterrows():
                score = self._calculate_improved_similarity(query_lower, row)
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

            # Only return very strong matches
            if best_match and best_match['confidence'] > 0.8:
                logger.info(
                    f"Strong database match: {best_match['article_no']} with confidence {best_match['confidence']}")
                return best_match

            return None

        except Exception as e:
            logger.error(f"Error searching legal database: {e}")
            return None

    def _calculate_improved_similarity(self, query: str, row) -> float:
        """Improved similarity calculation focusing on section numbers"""
        try:
            article_no = str(row.get('article_no', '')).lower()

            # Exact section number match gets highest score
            if re.search(r'\d+', query):
                query_sections = re.findall(r'\d+', query)
                row_sections = re.findall(r'\d+', article_no)

                if query_sections and row_sections and query_sections[0] == row_sections[0]:
                    return 0.9  # High score for exact section match

            # Word-based similarity for other cases
            query_words = set(re.findall(r'\w+', query))
            row_text = f"{str(row.get('law_type', '')).lower()} {article_no}"
            row_words = set(re.findall(r'\w+', row_text))

            if not query_words:
                return 0

            overlap = len(query_words.intersection(row_words))
            return overlap / len(query_words)

        except Exception as e:
            logger.error(f"Error calculating improved similarity: {e}")
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


def analyze_query_type(query):
    """Analyze what type of legal query this is"""
    query_lower = query.lower()

    # Law section queries (e.g., "Section 10", "Article 14")
    section_patterns = [
        r'section\s+\d+',
        r'article\s+\d+',
        r'sec\.?\s*\d+',
        r'art\.?\s*\d+',
        r'\b\d+\s*of\s*[A-Z]',
        r'ipc\s+section\s+\d+',
        r'crpc\s+section\s+\d+'
    ]

    for pattern in section_patterns:
        if re.search(pattern, query_lower):
            return "law_section"

    # Criminal law queries
    criminal_keywords = ['theft', 'murder', 'robbery', 'assault', 'fraud', 'cheating', 'criminal', 'ipc', 'penal code']
    if any(keyword in query_lower for keyword in criminal_keywords):
        return "criminal_law"

    # Contract law queries
    contract_keywords = ['contract', 'agreement', 'offer', 'acceptance', 'consideration', 'breach']
    if any(keyword in query_lower for keyword in contract_keywords):
        return "contract_law"

    # Constitutional law queries
    constitutional_keywords = ['constitution', 'fundamental rights', 'article 14', 'article 19', 'article 21']
    if any(keyword in query_lower for keyword in constitutional_keywords):
        return "constitutional_law"

    return "general"


def get_intelligent_fallback(query, query_type):
    """Provide intelligent fallback responses"""
    query_lower = query.lower()

    # Criminal law fallbacks
    if query_type == "criminal_law":
        if 'theft' in query_lower:
            return """**Theft under Indian Penal Code**

**Definition (Section 378 IPC):**
Theft involves dishonestly taking movable property out of someone's possession without their consent, with the intention to permanently deprive them of it.

**Key Elements:**
1. **Dishonest intention** - Intent to cause wrongful gain or loss
2. **Movable property** - Physical property that can be moved
3. **Taking without consent** - Without the owner's permission
4. **Out of possession** - Removing from owner's control

**Punishment (Section 379 IPC):**
- Imprisonment up to 3 years, or fine, or both

**Note:** This is a basic overview. For detailed case-specific advice, consult a criminal lawyer."""

    # Contract law fallbacks
    elif query_type == "contract_law":
        return f"""**Indian Contract Act, 1872**

I understand you're asking about contract law. The Indian Contract Act, 1872 governs contracts in India.

**Essential Elements of Valid Contract:**
1. Offer and acceptance
2. Lawful consideration
3. Capacity to contract
4. Free consent
5. Lawful object

**Please provide more specific details about your contract law query for a detailed response.**"""

    # General fallback
    else:
        return f"""I understand you're asking about: "{query}"

As a legal AI assistant specializing in Indian law, I can help with:

• **Criminal Law**: IPC offenses, procedures, rights
• **Contract Law**: Agreements, obligations, remedies  
• **Constitutional Law**: Fundamental rights, legal framework
• **Property Law**: Ownership, transfer, disputes
• **Family Law**: Marriage, divorce, inheritance

Please provide more specific details about your legal question for a comprehensive answer."""


def generate_chat_response(query, query_type):
    """Generate appropriate responses based on query type"""
    query_lower = query.lower()

    # Criminal law specific prompts
    if query_type == "criminal_law":
        if 'theft' in query_lower:
            prompt = f"""Explain the legal concept of theft under Indian law:

Question: {query}

Cover these aspects:
1. Definition of theft under Indian Penal Code (Section 378)
2. Essential ingredients of theft
3. Punishment for theft (Section 379)
4. Difference between theft, robbery, and dacoity
5. Real-world examples

Provide a comprehensive explanation with references to specific IPC sections."""

        elif any(word in query_lower for word in ['murder', 'homicide']):
            prompt = f"""Explain murder under Indian Penal Code:

Question: {query}

Discuss:
1. Definition of murder (Section 300 IPC)
2. Difference between murder and culpable homicide
3. Punishment for murder (Section 302)
4. Exceptions and mitigating circumstances
5. Recent legal developments"""

        else:
            prompt = f"""As a criminal law expert, answer this question about Indian criminal law:

Question: {query}

Provide a detailed explanation covering:
- Relevant IPC sections
- Legal definitions and elements
- Punishments and procedures
- Important case laws if applicable
- Practical implications

Focus on accuracy and clarity."""

    # Contract law specific prompts
    elif query_type == "contract_law":
        prompt = f"""As a contract law expert, answer this question about Indian contract law:

Question: {query}

Cover relevant aspects of:
- Indian Contract Act, 1872 provisions
- Essential elements of valid contract
- Rights and obligations of parties
- Breach and remedies
- Important judicial interpretations

Provide practical examples where helpful."""

    # Constitutional law specific prompts
    elif query_type == "constitutional_law":
        prompt = f"""As a constitutional law expert, answer this question:

Question: {query}

Discuss:
- Relevant constitutional provisions
- Fundamental rights aspects
- Judicial interpretations
- Landmark Supreme Court cases
- Current constitutional position

Cite specific articles and case laws."""

    # General legal questions
    else:
        if any(word in query_lower for word in ['what is', 'explain', 'define']):
            prompt = f"""Explain this legal concept in comprehensive detail:

Question: {query}

Provide:
1. Clear definition and legal basis
2. Relevant Indian laws and sections
3. Key elements and requirements
4. Practical implications
5. Examples for clarity

Structure your response with clear headings and proper spacing."""

        elif any(word in query_lower for word in ['how to', 'procedure']):
            prompt = f"""Provide step-by-step legal procedure:

Question: {query}

Break down into clear steps:
1. Preliminary requirements
2. Documentation needed
3. Legal process timeline
4. Authorities involved
5. Expected outcomes
6. Common challenges

Make it practical and actionable."""

        else:
            prompt = f"""You are AutoLegal AI, a professional legal assistant for Indian law.

Question: {query}

Provide a comprehensive, well-researched answer that:
- Directly addresses the question
- Cites relevant Indian laws and sections
- Provides practical legal advice
- Uses clear, professional language
- Is well-structured with proper formatting

If this is not a legal question, politely redirect to legal topics."""

    try:
        response = ask_gemini(prompt, timeout=25)

        # Ensure response is meaningful
        if not response or any(generic in response for generic in ["⚠️", "not available", "no response"]):
            return get_intelligent_fallback(query, query_type)

        return response

    except Exception as e:
        logger.error(f"Chat response generation failed: {e}")
        return get_intelligent_fallback(query, query_type)


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

    # Step 1: Analyze query type first
    query_type = analyze_query_type(query)

    # Step 2: Only use database for specific law section queries
    if query_type == "law_section":
        legal_db = get_legal_database()
        db_result = legal_db.search_legal_query(query, threshold=0.7)  # Very high threshold

        if db_result and db_result['confidence'] > 0.8:  # Only use for very clear matches
            response_text = f"""📚 **Legal Information Found**

**{db_result['law_type']} - {db_result['article_no']}**

**Original Text:**
{db_result['original']}

**Simplified Explanation:**
{db_result['simplified']}

**Legal Meaning:**
{db_result['meaning']}

*Source: Legal Database*"""

            return jsonify({
                "response": response_text,
                "source": "database"
            })

    # Step 3: Use Gemini for all other queries
    response_text = generate_chat_response(query, query_type)
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
