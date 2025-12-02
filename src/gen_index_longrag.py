import json
import re
import os
import time
import faiss
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path
from utils.embeddings import get_embedding_client

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)
model2path = config["model_path"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dataset and generate embeddings.")
    parser.add_argument("--dataset", help="Name of the dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"])
    parser.add_argument('--chunk_size', type=int, default=200, help="Minimum chunk size for splitting")
    parser.add_argument('--min_sentence', type=int, default=2, help="Minimum number of sentences in a chunk")
    parser.add_argument('--overlap', type=int, default=2, help="Number of overlapping sentences between chunks")
    parser.add_argument('--max_docs', type=int, default=None, help="Optional: limit number of documents to process (for testing)")
    return parser.parse_args()

def get_word_count(text):
    regEx = re.compile('[\W]')
    chinese_char_re = re.compile(r"([\u4e00-\u9fa5])")
    words = regEx.split(text.lower())
    word_list = []
    for word in words:
        if chinese_char_re.split(word):
            word_list.extend(chinese_char_re.split(word))
        else:
            word_list.append(word)
    return len([w for w in word_list if len(w.strip()) > 0])

def split_sentences(content, chunk_size, min_sentence, overlap):
    stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';']
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    sentences = re.split(split_pattern, content)
    
    if len(sentences) == 1:
        return sentences
    
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences) - 1, 2)]
    chunks = []
    temp_text = ''
    sentence_overlap_len = 0
    start_index = 0

    for i, sentence in enumerate(sentences):
        temp_text += sentence
        if get_word_count(temp_text) >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
            if i + 1 > overlap:
                sentence_overlap_len = sum([get_word_count(sentences[j]) for j in range(i+1-overlap, i+1)])
            if chunks:
                if start_index > overlap:
                    start_index -= overlap
            chunk_text = ''.join(sentences[start_index:i+1])
            if not chunks:
                chunks.append(chunk_text)
            elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)
            temp_text = ''
            start_index = i + 1
    
    return chunks

def process_data(file_path, chunk_size, min_sentence, overlap, save_path, max_docs=None):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_rawid = {}
    processed_chunks = []

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing data"):
        if max_docs is not None and idx >= max_docs:
            break
        content = item.get("paragraph_text") or item.get("ch_content") or item.get("ch_contenn")
        chunks = split_sentences(content, chunk_size, min_sentence, overlap)
        for i, chunk in enumerate(chunks):
            id_to_rawid[len(processed_chunks) + i] = idx
        processed_chunks.extend(chunks)
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/chunks.json", "w", encoding='utf-8') as fout:
        json.dump(processed_chunks, fout, ensure_ascii=False)
    with open(f"{save_path}/id_to_rawid.json", "w", encoding='utf-8') as fout:
        json.dump(id_to_rawid, fout, ensure_ascii=False)
    
    return processed_chunks

def calculate_embeddings(content, embedding_client, vector_store_path, batch_size=32):
    # embedding_client: EmbeddingClient instance with .encode(list[str]) -> np.array
    # Note: batch_size=32 works well with Ollama; larger sizes may timeout
    import numpy as np

    # Generate embeddings in batches to avoid memory pressure
    all_embeddings = []
    total_batches = (len(content) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(content), batch_size), total=total_batches, desc="Embedding batches"):
        batch = content[i:i+batch_size]
        emb = embedding_client.encode(batch)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vector_store_path)

def main():
    args = parse_arguments()
    # Resolve project root relative to this file so script works from any cwd
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    save_path = str(PROJECT_ROOT / 'data' / 'corpus' / 'processed' / f"{args.chunk_size}_{args.min_sentence}_{args.overlap}" / args.dataset)
    vector_store_path = f"{save_path}/vector.index"

    print("Starting data processing...")
    raw_file = str(PROJECT_ROOT / 'data' / 'corpus' / 'raw' / f"{args.dataset}.json")
    content = process_data(raw_file, args.chunk_size, args.min_sentence, args.overlap, save_path, max_docs=args.max_docs)
    
    print("Calculating embeddings...")
    start_time = time.time()
    emb_override = os.getenv("EMBEDDING_MODEL_OVERRIDE")
    emb_model = emb_override or model2path["emb_model"]
    embedding_client = get_embedding_client(emb_model)
    calculate_embeddings(content, embedding_client, vector_store_path)
    end_time = time.time()
    
    print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()