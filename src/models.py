import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq
from config import GROQ_MODEL

@st.cache_resource
def load_local_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embedding_model, reranker

def get_local_llm():
    return load_local_llm()

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_groq(system_prompt, user_prompt, max_tokens=512, temperature=0.2):
    client = get_groq_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        st.warning(f"Groq API error: {exc}. Falling back to local model.")
        return None

@st.cache_resource
def connect_qdrant():
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error(
            "❌ Missing Qdrant credentials. "
            "Please create a `.env` file with QDRANT_URL and QDRANT_API_KEY set."
        )
        st.stop()

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False
    )
    return client
