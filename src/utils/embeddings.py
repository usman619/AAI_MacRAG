import os
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

import requests
try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    genai = None
    _HAS_GENAI = False


class EmbeddingClient:
    def encode(self, texts):
        raise NotImplementedError()


class HFEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name_or_path):
        if not _HAS_ST:
            raise RuntimeError("sentence_transformers not installed; install with `pip install sentence-transformers` to use HF embeddings")
        # allow local path or HF repo id
        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, texts, batch_size=32):
        # returns numpy array (n, dim)
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings)


class OllamaEmbeddingClient(EmbeddingClient):
    """Client for Ollama embedding models using the /api/embed endpoint."""

    def __init__(self, base_url: str, model: str):
        # base_url should be like http://localhost:11434
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embed_url = f"{self.base_url}/api/embed"

    def encode(self, texts, batch_size=64):
        all_embs = []
        headers = {"Content-Type": "application/json"}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Ollama /api/embed accepts {"model": "...", "input": [...]}
            payload = {"model": self.model, "input": batch}
            resp = requests.post(self.embed_url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Ollama returns {"embeddings": [[...], [...]]} for batch input
            if "embeddings" in data:
                batch_emb = data["embeddings"]
            elif "embedding" in data:
                # Single embedding case (older Ollama versions or single input)
                emb = data["embedding"]
                if len(emb) > 0 and isinstance(emb[0], (list, tuple)):
                    batch_emb = emb
                else:
                    batch_emb = [emb]
            else:
                raise RuntimeError(f"Unexpected Ollama response format: {data}")

            all_embs.extend(batch_emb)

        return np.array(all_embs)


class HTTPEmbeddingClient(EmbeddingClient):
    def __init__(self, url):
        self.url = url

    def encode(self, texts, batch_size=64):
        all_embs = []
        # Try a few common payload keys used by embedding services
        payload_variants = [
            ("inputs", lambda b: {"inputs": b}),
            ("input", lambda b: {"input": b}),
            ("text", lambda b: {"text": b}),
            ("texts", lambda b: {"texts": b}),
        ]

        headers = {"Content-Type": "application/json"}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            last_exc = None
            resp_json = None

            # Try each payload shape until one succeeds
            for name, make_payload in payload_variants:
                try:
                    payload = make_payload(batch)
                    resp = requests.post(self.url, json=payload, headers=headers, timeout=60)
                    # If the server returns 404/405/400 for this payload shape, try the next
                    if resp.status_code >= 400 and resp.status_code < 500:
                        last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                        continue
                    resp.raise_for_status()
                    resp_json = resp.json()
                    break
                except Exception as e:
                    last_exc = e
                    continue

            if resp_json is None:
                # No payload shape worked
                raise RuntimeError(f"Failed to get embeddings from {self.url}; last error: {last_exc}")

            # Normalize the response into a list of embeddings
            batch_emb = None
            if isinstance(resp_json, dict):
                if "embeddings" in resp_json:
                    batch_emb = resp_json["embeddings"]
                elif "data" in resp_json and isinstance(resp_json["data"], list):
                    # HuggingFace-like: {'data': [{'embedding': [...]}, ...]}
                    batch_emb = [d.get("embedding") or d.get("vector") for d in resp_json["data"]]
                elif "embedding" in resp_json:
                    emb = resp_json["embedding"]
                    # If it's a list of lists -> already batched
                    if len(emb) > 0 and isinstance(emb[0], (list, tuple)):
                        batch_emb = emb
                    else:
                        # single embedding for the whole batch -> wrap
                        batch_emb = [emb]
                else:
                    # Try to extract any list-of-number sequences as a last-ditch attempt
                    candidates = []
                    for v in resp_json.values():
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                            candidates.append(v)
                    if candidates:
                        batch_emb = candidates

            elif isinstance(resp_json, list):
                # Could be a plain list of embedding vectors
                batch_emb = resp_json

            if batch_emb is None:
                raise RuntimeError(f"Unrecognized embedding response format from {self.url}: {resp_json}")

            all_embs.extend(batch_emb)

        return np.array(all_embs)


def get_embedding_client(model_name_or_path_or_url: str):
    """Return an EmbeddingClient.

    Behavior (checked in order):
    1. If OLLAMA_BASE_URL env var is set (and optionally OLLAMA_EMBED_MODEL), use OllamaEmbeddingClient.
    2. If EMBEDDING_SERVICE_URL env var is set, use HTTPEmbeddingClient pointing there.
    3. If the provided string starts with "ollama:" (e.g. "ollama:nomic-embed-text"), use Ollama at localhost:11434.
    4. If the provided string is an HTTP(S) URL, use HTTPEmbeddingClient.
    5. If it's an existing local path, use HFEmbeddingClient (SentenceTransformer).
    6. If it looks like a Gemini model name, use GeminiEmbeddingClient.
    7. Otherwise, use HFEmbeddingClient (assumes HF Hub model id).
    """
    # 1. Ollama via env vars
    ollama_base = os.getenv("OLLAMA_BASE_URL")
    if ollama_base:
        ollama_model = os.getenv("OLLAMA_EMBED_MODEL", model_name_or_path_or_url)
        return OllamaEmbeddingClient(base_url=ollama_base, model=ollama_model)

    # 2. Generic HTTP endpoint
    env_url = os.getenv("EMBEDDING_SERVICE_URL")
    if env_url:
        return HTTPEmbeddingClient(env_url)

    # 3. Ollama prefix in model name (e.g. "ollama:nomic-embed-text")
    if model_name_or_path_or_url.startswith("ollama:"):
        model = model_name_or_path_or_url.split(":", 1)[1]
        return OllamaEmbeddingClient(base_url="http://localhost:11434", model=model)

    # 4. If it's a URL
    if model_name_or_path_or_url.startswith("http://") or model_name_or_path_or_url.startswith("https://"):
        return HTTPEmbeddingClient(model_name_or_path_or_url)

    # If it's a local path
    p = Path(model_name_or_path_or_url)
    if p.exists():
        return HFEmbeddingClient(str(p))

    # Gemini embeddings (SDK) when model name indicates Gemini
    if _HAS_GENAI and ("gemini" in model_name_or_path_or_url.lower() or model_name_or_path_or_url.startswith("gemini-")):
        # Ensure SDK configured (configure now if key available)
        gem_key = os.getenv("GEMINI_API_KEY")
        if gem_key:
            try:
                genai.configure(api_key=gem_key)
            except Exception:
                # continue; SDK may already be configured or configuration may differ
                pass
        class GeminiEmbeddingClient(EmbeddingClient):
            def __init__(self, model):
                self.model = model

            def encode(self, texts, batch_size=64):
                all_embs = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    # Use the installed SDK's embed_content helper which may be
                    # named differently across versions (embed_content/embed_content_async).
                    # Call embed_content with the batch (it will internally batch requests
                    # if needed) and normalize the response.
                    try:
                        resp = genai.embed_content(model=self.model, content=batch)
                    except Exception:
                        # Fallback: try single-item calls
                        batch_emb = []
                        for doc in batch:
                            r = genai.embed_content(model=self.model, content=doc)
                            if isinstance(r, dict) and 'embedding' in r:
                                batch_emb.append(r['embedding'])
                            else:
                                batch_emb.append(r)
                        all_embs.extend(batch_emb)
                        continue

                    # Normalize response shapes
                    # Expected shapes:
                    # - {'embedding': [float,...]} for single input
                    # - {'embedding': [[...], [...], ...]} for batch input
                    if isinstance(resp, dict) and 'embedding' in resp:
                        emb = resp['embedding']
                        # If emb is a list of lists -> batch
                        if len(emb) > 0 and isinstance(emb[0], (list, tuple)):
                            batch_emb = emb
                        else:
                            # single embedding returned
                            batch_emb = [emb]
                    else:
                        # As a last resort, try to interpret resp as an iterable of embeddings
                        batch_emb = list(resp)

                    all_embs.extend(batch_emb)

                import numpy as _np
                return _np.array(all_embs)

        return GeminiEmbeddingClient(model_name_or_path_or_url)

    # Fallback to HF model id
    return HFEmbeddingClient(model_name_or_path_or_url)
