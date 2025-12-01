      
import json
import requests
import re
# from zhipuai import ZhipuAI
from openai import OpenAI
import backoff
import time
from openai import OpenAI
import httpx
import yaml
from pathlib import Path

# Load configuration relative to the repository root (two levels up from this file)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)["api"]
from dotenv import load_dotenv
load_dotenv()
import os
from utils.gemini_handler import get_gemini_response

def glm(prompt,model,max_tokens):
    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=config["zp_key"]) 
        prompt=[{"role": "user", "content":prompt}]
        response = client.chat.completions.create(
        model=model,
        temperature = 0.99,
        messages=prompt,
        max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
            print(f"An error occurred: {e}")  
            time.sleep(0.5)
            return None

def gemini(prompt, model, max_tokens, temperature):
    # Use SDK helper which reads GEMINI_API_KEY from environment/.env.
    try:
        return get_gemini_response(prompt, model_name=model, temperature=temperature)
    except Exception as e:
        print(f"An error occurred while calling gemini helper: {e}")
        return "None"
    


def gpt(prompt,model,max_tokens,temperature): 
    try:
        client = OpenAI(
            api_key=config["openai_key"],
        )
        message = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(model= model, messages= message, max_tokens=max_tokens, temperature=temperature)      
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"An error occurred: {e}")  
        time.sleep(1)
    return "None"

# Remove duplicate sentences to prevent GPT API from refusing to respond due to repeated input

def remove_consecutive_repeated_sentences(text, threshold=5):
    sentences = re.split(r'([。！？,，])', text)
    
    cleaned_sentences = []
    current_sentence = None
    count = 0

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            delimiter = sentences[i + 1]
        else:
            delimiter = ''

        if sentence == current_sentence:
            count += 1
        else:
            if count >= threshold:
                cleaned_sentences.append(current_sentence + delimiter)
            elif current_sentence:
                cleaned_sentences.extend([current_sentence + delimiter] * count)
            current_sentence = sentence
            count = 1

    if count >= threshold:
        cleaned_sentences.append(current_sentence + delimiter)
    else:
        cleaned_sentences.extend([current_sentence + delimiter] * count)
    
    cleaned_text = ''.join(cleaned_sentences)
    return cleaned_text


@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_api(prompt,model,max_new_tokens, temperature = 1):
    if "glm" in model:
        res=glm(prompt,model, max_new_tokens)
    elif "gpt" in model:
        res=gpt(prompt,model,max_new_tokens, temperature)
        if not res:
            prompt=remove_consecutive_repeated_sentences(prompt)
            res=gpt(prompt,model,max_new_tokens)
    elif "gemini" in model:
        # Prefer SDK-based helper which reads GEMINI_API_KEY from env/.env
        try:
            res = get_gemini_response(prompt, model_name=model, temperature=temperature)
            # fallback to HTTP-based gemini() if helper failed or returned an error marker
            if not res or (isinstance(res, str) and res.startswith("Error:")):
                res = gemini(prompt, model, max_new_tokens, temperature)
        except Exception:
            res = gemini(prompt, model, max_new_tokens, temperature)
    assert res != None
    return res

if __name__ == "__main__":
    # Safe runtime checks: only call remote APIs if the corresponding keys/SDKs are available.
    print("Configuration quick-check:")
    print("  zp_key present:", bool(config.get("zp_key")))
    print("  openai_key present:", bool(config.get("openai_key")))
    from dotenv import load_dotenv
    load_dotenv()
    import os
    print("  GEMINI_API_KEY present:", bool(os.getenv("GEMINI_API_KEY")))

    # Gemini SDK helper
    try:
        from utils.gemini_handler import _HAS_GENAI as _HAS_GENAI, _CONFIGURED as _GEMINI_CONFIG
    except Exception:
        _HAS_GENAI = False
        _GEMINI_CONFIG = False

    if _HAS_GENAI and _GEMINI_CONFIG:
        print("Running a tiny Gemini test...")
        print(get_gemini_response("Say hello in one sentence.", model_name="gemini-2.5-flash", temperature=0))
    else:
        print("Skipping Gemini test: SDK or GEMINI_API_KEY not configured.")

    if config.get("zp_key"):
        print("Running a tiny GLM/Zhipuai test...")
        try:
            print(glm("Hello", "glm-4", 100))
        except Exception as e:
            print("GLM test failed:", e)
    else:
        print("Skipping GLM test: zp_key not configured.")

    if config.get("openai_key"):
        print("Running a tiny OpenAI test...")
        try:
            print(gpt("Hello", "gpt-4o", 100, 0))
        except Exception as e:
            print("OpenAI test failed:", e)
    else:
        print("Skipping OpenAI test: openai_key not configured.")
    



    
