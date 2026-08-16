import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import trange, tqdm

# Optional: import for different backends
from sentence_transformers import SentenceTransformer

# For GTR-T5 (T5 family)
from transformers import AutoTokenizer, AutoModel

# For OpenAI API
try:
    import openai
except ImportError:
    openai = None

# ===========================
# Config - CHOOSE YOUR MODEL
# ===========================
BACKEND = "sentence-transformers"   # Options: 'sentence-transformers', 'openai'
MODEL_NAME = "all-mpnet-base-v2"  # Examples: "intfloat/e5-base-v2", "nomic-ai/nomic-embed-text-v1", "gtr-t5-base", "text-embedding-3-large", "all-mpnet-base-v2", GeoGPT/GeoEmbedding
USE_FP16 = True                     # Only used for SentenceTransformers
NORMALIZE = True                    # Only used for SentenceTransformers

# If using OpenAI, set your API key here or via environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")  # Fill in or set env var

# ===========================
# Data Loading
# ===========================
def build_placeid_to_text(df_merged, text_col="text_input"):
    m = df_merged[text_col].notna() & (df_merged[text_col].astype(str).str.strip() != "")
    df_use = df_merged.loc[m, ["place_id", text_col]].drop_duplicates("place_id")
    return df_use.reset_index(drop=True)

# ===========================
# SentenceTransformer Backend
# ===========================
@torch.no_grad()
def encode_texts_sentence_transformers(texts, model_name, batch_size=256, use_fp16=True, normalize=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    if use_fp16 and device == "cuda":
        model = model.half()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize
    )
    return embs

# ===========================
# OpenAI Backend
# ===========================
def encode_texts_openai(texts, model="text-embedding-3-large", batch_size=96, api_key=None):
    assert openai is not None, "openai python package is required for this backend!"
    openai.api_key = api_key or OPENAI_API_KEY
    all_embeds = []
    for i in trange(0, len(texts), batch_size, desc="Encoding with OpenAI API"):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeds = [item['embedding'] for item in response.data]
        all_embeds.extend(batch_embeds)
    return np.array(all_embeds, dtype=np.float32)

# ===========================
# Unified Encoding Interface
# ===========================
def encode_texts(texts, backend, model_name, **kwargs):
    if backend == "sentence-transformers":
        return encode_texts_sentence_transformers(texts, model_name=model_name, **kwargs)
    elif backend == "openai":
        return encode_texts_openai(texts, model=model_name, **kwargs)
    else:
        raise NotImplementedError(f"Unknown backend: {backend}")

# ===========================
# Save as PyTorch Dict
# ===========================
def save_dict(placeid2emb: dict, out_path="poi_text_embeds.pt"):
    save_dict = {pid: torch.as_tensor(vec, dtype=torch.float32) for pid, vec in placeid2emb.items()}
    torch.save(save_dict, out_path)
    return out_path

# ===========================
# MAIN SCRIPT
# ===========================
if __name__ == "__main__":
    # Load your dataframes as before
    df_text = pd.read_csv("/mnt/disk/data/POI_data/Safegraph/pois_Houston_text_description_cleaned.csv")
    df_veraset = pd.read_parquet("/mnt/disk/data/trajfm_veraset_splits/veraset/Visits/Houston/whole_veraset_processed.parquet")
    df_veraset = df_veraset[["place_id", "safegraph.place_id"]].drop_duplicates()
    df_text = df_text[df_text['safegraph_place_id'].isin(df_veraset['safegraph.place_id'])]
    df_merged = pd.merge(df_text, df_veraset, left_on='safegraph_place_id', right_on='safegraph.place_id', how='inner')
    df_merged = df_merged.drop(columns=['safegraph_place_id'])

    df_use = build_placeid_to_text(df_merged, text_col="text_input")
    texts = df_use["text_input"].astype(str).tolist()
    place_ids = df_use["place_id"].tolist()

    # E5 models expect "query: ..." prefix
    if "e5" in MODEL_NAME.lower():
        texts = [f"query: {t}" for t in texts]

    # Choose backend based on model name
    if BACKEND == "sentence-transformers":
        print(f"Using SentenceTransformer: {MODEL_NAME}")
        Z = encode_texts(texts, backend=BACKEND, model_name=MODEL_NAME, batch_size=256, use_fp16=USE_FP16, normalize=NORMALIZE)
    elif BACKEND == "openai":
        print(f"Using OpenAI: {MODEL_NAME}")
        Z = encode_texts(texts, backend=BACKEND, model_name=MODEL_NAME, batch_size=96, api_key=OPENAI_API_KEY)
    else:
        raise NotImplementedError("Unknown backend")

    placeid2emb = {pid: Z[i] for i, pid in enumerate(place_ids)}
    out_path = f"/home/maria/Object-based-FM/embeddings/Houston/poi_text_embeds_{os.path.basename(MODEL_NAME)}.pt"
    save_dict(placeid2emb, out_path=out_path)
    print(f"Saved {len(placeid2emb)} embeddings to {out_path}")
