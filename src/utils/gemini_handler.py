import os
import time
from dotenv import load_dotenv

load_dotenv()

# Try to import the official Google Generative AI SDK. If it's not installed, keep
# the module importable and provide a helpful message at runtime.
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    _HAS_GENAI = True
except Exception:
    genai = None
    # Use a generic Exception type as a placeholder for ResourceExhausted
    ResourceExhausted = Exception
    _HAS_GENAI = False

# Configure the API key once when the module loads.
# Prefer using the environment variable GEMINI_API_KEY (or a .env file for local dev).
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if _GEMINI_KEY and _HAS_GENAI:
    genai.configure(api_key=_GEMINI_KEY)
    _CONFIGURED = True
else:
    if not _HAS_GENAI:
        print("WARNING: google.generativeai SDK not installed. Install it with `pip install google-generative-ai` to use Gemini.")
    else:
        print("WARNING: GEMINI_API_KEY not found in environment variables.\n" \
              "Set GEMINI_API_KEY in your shell or in a .env file (local dev).")
    _CONFIGURED = False

def get_gemini_response(prompt, model_name="gemini-2.5-flash", temperature=0):
    """
    Sends a prompt to Gemini and handles rate limits.
    """
    if not _CONFIGURED:
        if not _HAS_GENAI:
            return "Error: google.generativeai SDK not installed. Install with `pip install google-generative-ai`."
        return "Error: GEMINI_API_KEY not configured."

    model = genai.GenerativeModel(model_name)
    
    # Retry logic in case you hit the free tier rate limit
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text.strip()
            
        except ResourceExhausted:
            wait_time = 2 ** attempt # Exponential backoff: 1s, 2s, 4s...
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return ""
            
    return "Error: Failed to get response after retries."