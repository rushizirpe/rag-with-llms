# src/config.py

import os
from dotenv import load_dotenv
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

PERSIST_DIR = "./db"
DATA_DIR = "data"

def configure_settings():
    Settings.llm = HuggingFaceInferenceAPI(
        model_name="google/gemma-1.1-7b-it",
        tokenizer_name="google/gemma-1.1-7b-it",
        context_window=3000,
        token=os.getenv("HF_TOKEN"),
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1},
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)