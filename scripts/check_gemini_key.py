#!/usr/bin/env python3
"""Simple script to verify Gemini SDK and GEMINI_API_KEY configuration.

Run from repository root:
  python scripts/check_gemini_key.py

It will print whether the SDK is installed and whether the API key was found. If both are present, it will send a tiny test prompt.
"""
import sys
from dotenv import load_dotenv
load_dotenv()
import os

# ensure src is on path
sys.path.insert(0, "src")

try:
    from utils.gemini_handler import get_gemini_response, _HAS_GENAI, _CONFIGURED
except Exception as e:
    print(f"Error importing gemini helper: {e}")
    _HAS_GENAI = False
    _CONFIGURED = False

print(f"google.generativeai installed: {_HAS_GENAI}")
print(f"GEMINI_API_KEY configured: {_CONFIGURED}")

if not _HAS_GENAI:
    print("Install the SDK: pip install google-generative-ai")
    sys.exit(1)

if not _CONFIGURED:
    print("Set GEMINI_API_KEY in your environment or create a .env file at the project root with GEMINI_API_KEY=...")
    sys.exit(1)

print("Sending a tiny test prompt to Gemini...")
res = get_gemini_response("Say hello in one sentence.", model_name="gemini-2.5-flash", temperature=0)
print("Response:", res)
