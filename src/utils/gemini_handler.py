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

# Track last request time for rate limiting
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 7  # seconds between requests (free tier: 10 requests/minute)

def get_gemini_response(prompt, model_name="gemini-2.5-flash", temperature=0):
    """
    Sends a prompt to Gemini and handles rate limits.
    Includes built-in rate limiting for free tier (10 requests/minute).
    """
    global _last_request_time
    
    if not _CONFIGURED:
        if not _HAS_GENAI:
            return "Error: google.generativeai SDK not installed. Install with `pip install google-generative-ai`."
        return "Error: GEMINI_API_KEY not configured."

    # Rate limiting: ensure minimum interval between requests
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        wait_time = _MIN_REQUEST_INTERVAL - elapsed
        time.sleep(wait_time)
    
    model = genai.GenerativeModel(model_name)
    
    # Retry logic in case you hit the free tier rate limit
    max_retries = 5
    empty_response_count = 0
    max_empty_responses = 2  # Only retry empty responses twice
    
    for attempt in range(max_retries):
        try:
            _last_request_time = time.time()
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            # Handle cases where response has no valid parts
            if response.candidates and response.candidates[0].content.parts:
                text = response.text.strip()
                if text:
                    return text
                else:
                    # Empty text response
                    empty_response_count += 1
                    if empty_response_count >= max_empty_responses:
                        return "None"  # Give up on empty responses
                    wait_time = 2 ** attempt
                    print(f"Empty response, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            else:
                # No valid parts in response (content filtering, etc.)
                empty_response_count += 1
                if empty_response_count >= max_empty_responses:
                    return "None"  # Give up
                wait_time = 2 ** attempt
                print(f"No valid parts in response, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            
        except ResourceExhausted:
            wait_time = 2 ** attempt + 10  # Add extra time for rate limits
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                wait_time = 2 ** attempt + 10
                print(f"Rate limit detected. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error calling Gemini: {e}")
                # For other errors, try once more after a short wait
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "None"
            
    return "None"