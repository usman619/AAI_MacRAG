import os
import time
from dotenv import load_dotenv

load_dotenv()

# Try to import the official Google Generative AI SDK. If it's not installed, keep
# this module importable and provide a helpful message at runtime.
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    genai = None
    _HAS_GENAI = False

# Configure once at module import using GEMINI_API_KEY
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if _GEMINI_KEY and _HAS_GENAI:
    genai.configure(api_key=_GEMINI_KEY)
    _CONFIGURED = True
else:
    if not _HAS_GENAI:
        print("WARNING: google.generativeai SDK not installed. Install it with `pip install google-generative-ai` to use Gemini.")
    else:
        print("WARNING: GEMINI_API_KEY not found. Set it in your environment or .env file.")
    _CONFIGURED = False


def query_gemini(prompt, model_name="gemini-2.5-pro"):
    """Send prompt to Gemini. Returns text or an empty string on error.

    Note: This helper expects the module to be configured at import time with GEMINI_API_KEY.
    """
    if not _CONFIGURED:
        if not _HAS_GENAI:
            print("Gemini SDK not installed. Install with: pip install google-generative-ai")
        else:
            print("Gemini API key not configured. Returning empty response.")
        return ""

    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        time.sleep(2)
        return ""  # or retry logic