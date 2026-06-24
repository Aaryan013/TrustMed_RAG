import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi
from groq import Groq
import torch
import concurrent.futures

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL_ANSWER = "I could not find reliable medical information in the retrieved documents."
QUERY_ALIGNMENT_THRESHOLD = 0.55

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="MediTrust AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better formatting
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0e1117;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        color: #f8fafc;
    }
    .user-message {
        background-color: #1e293b;
        border-left: 5px solid #3b82f6;
    }
    .bot-message {
        background-color: #064e3b;
        border-left: 5px solid #10b981;
    }
    .status-dot {
        height: 10px; width: 10px; background-color: #10b981; border-radius: 50%; display: inline-block; margin-right: 8px;
    }
    .status-dot-warning {
        height: 10px; width: 10px; background-color: #f59e0b; border-radius: 50%; display: inline-block; margin-right: 8px;
    }
    .source-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        transition: box-shadow 0.2s ease;
        color: #f8fafc;
    }
    .source-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    .source-title {
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .source-link {
        font-size: 0.85rem;
        color: #60a5fa;
        text-decoration: none;
    }
    .high-confidence { color: #10b981; font-weight: bold; }
    .medium-confidence { color: #f59e0b; font-weight: bold; }
    .low-confidence { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- CACHING MODELS & DATA ---

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

@st.cache_data
def load_data_and_bm25():
    try:
        chunked_df = pd.read_csv("chunked_medical_dataset.csv")
    except FileNotFoundError:
        return None, None
        
    corpus = chunked_df["context"].tolist()
    tokenized_corpus = [str(doc).lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return chunked_df, bm25

# Initialize components
with st.spinner("Initializing models... (This may take a minute on first run)"):
    embedding_model, reranker = load_models()
    qdrant_client = connect_qdrant()
    chunked_df, bm25 = load_data_and_bm25()

if chunked_df is None:
    st.error("Data not found. Please run `python data_prep.py` first to generate the chunked dataset.")
    st.stop()


# --- RAG CORE FUNCTIONS ---

import time

def qdrant_search(query, top_k=10, retries=3):
    query_embedding = embedding_model.encode(query).tolist()
    
    for attempt in range(retries):
        try:
            results = qdrant_client.query_points(
                collection_name="meditrust_rag_v2",
                query=query_embedding,
                limit=top_k
            )
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1) # Wait a second before retrying
            else:
                raise e

    formatted_results = []
    for r in results.points:
        formatted_results.append({
            "score": r.score,
            "topic": r.payload.get("topic"),
            "focus": r.payload.get("focus"),
            "qtype": r.payload.get("qtype"),
            "question": r.payload.get("question"),
            "url": r.payload.get("url"),
            "context": r.payload.get("context"),
            "chunk_id": r.payload.get("chunk_id")
        })
    return formatted_results

def bm25_search(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": scores[idx],
            "topic": chunked_df.iloc[idx]["topic"],
            "question": chunked_df.iloc[idx]["question"],
            "context": chunked_df.iloc[idx]["context"],
            "chunk_id": chunked_df.iloc[idx]["chunk_id"],
            "url": chunked_df.iloc[idx].get("url", "")
        })
    return results

def hybrid_search(query, top_k=10):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dense = executor.submit(qdrant_search, query, top_k)
        future_sparse = executor.submit(bm25_search, query, top_k)
        
        dense_results = future_dense.result()
        sparse_results = future_sparse.result()
    
    dense_scores = np.array([r["score"] for r in dense_results])
    if len(dense_scores) > 0:
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
    
    bm25_scores = np.array([r["score"] for r in sparse_results])
    if len(bm25_scores) > 0:
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
    combined_scores = defaultdict(float)
    combined_docs = {}
    
    for r, score in zip(dense_results, dense_scores):
        key = str(r["context"])
        combined_scores[key] += 0.6 * float(score)
        combined_docs[key] = r
        
    for r, score in zip(sparse_results, bm25_scores):
        key = str(r["context"])
        combined_scores[key] += 0.4 * float(score)
        if key not in combined_docs:
            combined_docs[key] = r
            
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for context, score in ranked[:top_k]:
        doc = combined_docs[context]
        doc["hybrid_score"] = score
        final_results.append(doc)
    return final_results

def rerank_results(query, retrieved_docs, top_k=3):
    pairs = [(query, str(doc["context"])) for doc in retrieved_docs]
    if not pairs:
        return []
        
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
        
    reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def build_context(docs):
    contexts = []
    for doc in docs:
        contexts.append(f"Topic: {doc.get('topic', 'Unknown')}\nFocus: {doc.get('focus', 'Unknown')}\nQuestion Type: {doc.get('qtype', 'Unknown')}\n\n{doc.get('context', '')}")
    return "\n\n".join(contexts)



def make_refusal_response(query):
    return {
        "query": query,
        "answer": REFUSAL_ANSWER,
        "confidence": "Low",
        "sources": [],
        "retrieval_explanations": [],
    }

def make_out_of_domain_response(query):
    return {
        "query": query,
        "answer": "I am a medical AI assistant and can only answer questions related to health and medicine. Please ask me a medical question.",
        "confidence": "Low",
        "sources": [],
        "retrieval_explanations": [],
    }


def is_out_of_domain(query, conversation_history=""):
    """Detect non-medical questions before retrieval."""
    system_prompt = """You classify questions for a medical Q&A assistant.
Reply with exactly one word: MEDICAL or NON_MEDICAL.

Use the conversation history when the question is a follow-up (e.g. "What are its types?").
MEDICAL includes: symptoms, diseases, treatments, medications, anatomy, mental health, nutrition for health, public health, clinical tests.
NON_MEDICAL includes: sports, politics, entertainment, programming, shopping, recipes (non-health), weather, general trivia, business, travel."""

    user_prompt = query
    if conversation_history.strip():
        user_prompt = f"""Conversation:
{conversation_history}

Question: {query}"""

    result = call_groq(system_prompt, user_prompt, max_tokens=8, temperature=0.0)
    if result:
        normalized = result.strip().upper()
        if "NON_MEDICAL" in normalized or normalized == "NON":
            return True
        if "MEDICAL" in normalized:
            return False

    if conversation_history.strip():
        return False

    query_vec = embedding_model.encode(query, normalize_embeddings=True)
    medical_anchors = [
        "What are the symptoms of diabetes?",
        "What causes high blood pressure?",
        "How is pneumonia treated?",
    ]
    non_medical_anchors = [
        "Who won the cricket match?",
        "Write a Python sorting program.",
        "What is the weather in New York today?",
    ]

    medical_embs = embedding_model.encode(medical_anchors, normalize_embeddings=True)
    non_medical_embs = embedding_model.encode(non_medical_anchors, normalize_embeddings=True)

    medical_sim = max(float(np.dot(query_vec, emb)) for emb in medical_embs)
    non_medical_sim = max(float(np.dot(query_vec, emb)) for emb in non_medical_embs)
    
    return non_medical_sim > medical_sim


def is_query_aligned_with_retrieval(query, retrieved_docs):
    """Ensure retrieved docs match the search query used for retrieval."""
    if not retrieved_docs:
        return False

    # Use precomputed scores from reranker instead of recalculating
    top_score = float(max([doc.get("rerank_score", 0) for doc in retrieved_docs[:3]]))
    return top_score >= QUERY_ALIGNMENT_THRESHOLD


def extract_sources(retrieved_docs):
    sources = []
    seen = set()
    for doc in retrieved_docs:
        url = doc.get("url", "")
        source = {
            "topic": doc.get("topic", ""),
            "focus": doc.get("focus", ""),
            "question": doc.get("question", ""),
            "url": url
        }
        key = url if url else str(doc.get("chunk_id", np.random.rand()))
        if key not in seen:
            sources.append(source)
            seen.add(key)
    return sources

def compute_confidence(retrieved_docs):
    if not retrieved_docs:
        return "Low"
    avg_score = np.mean([doc.get("rerank_score", 0) for doc in retrieved_docs])
    if avg_score > 0.95:
        return "High"
    elif avg_score > 0.85:
        return "Medium"
    else:
        return "Low"

def is_low_confidence(retrieved_docs):
    if not retrieved_docs:
        return True
    avg_score = np.mean([doc.get("rerank_score", 0) for doc in retrieved_docs])
    return avg_score < 0.80

def trustworthy_rag(query):
    retrieved_docs = hybrid_search(query, top_k=10)
    reranked_docs = rerank_results(query, retrieved_docs, top_k=3)

    if is_low_confidence(reranked_docs) or not is_query_aligned_with_retrieval(query, reranked_docs):
        return make_refusal_response(query)
        
    context = build_context(reranked_docs)

    system_prompt = """You are MediTrust AI, a careful medical information assistant.

Rules:
- Answer ONLY using the provided medical context.
- Be concise, clear, and well-formatted in Markdown.
- Structure responses with:
  1. A one-sentence direct answer
  2. **Key points** as bullet points (3-5 items max)
  3. A brief **Note** that this is informational, not medical advice
- Do not invent facts, diagnoses, or treatments.
- If the context does not support an answer, reply exactly:
"I could not find reliable medical information in the retrieved documents."
"""

    user_prompt = f"""Medical Context:
{context}

Question:
{query}"""

    answer = call_groq(system_prompt, user_prompt, max_tokens=450, temperature=0.2)

    if answer is None:
        tokenizer, model = get_local_llm()
        prompt = f"""You are a helpful medical assistant.
Answer the user's question ONLY using the provided medical context.
If the answer is not present in the context, say:
"I could not find reliable medical information in the retrieved documents."

Medical Context:
{context}

Question:
{query}

Answer:
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    sources = extract_sources(reranked_docs)
    confidence = compute_confidence(reranked_docs)
    
    explanations = []
    for doc in reranked_docs:
        explanations.append({
            "topic": doc.get("topic", ""),
            "question": doc.get("question", ""),
            "rerank_score": round(doc.get("rerank_score", 0), 4),
            "context_preview": str(doc.get("context", ""))[:300]
        })
        
    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "retrieval_explanations": explanations
    }

# --- PHASE 1: CONVERSATIONAL MEMORY & QUERY REWRITING ---

def build_conversation_history(messages, max_turns=3):
    """
    Extract the last N human+assistant exchange pairs from session state
    and format them as a readable conversation transcript for the rewriter.
    """
    history_lines = []
    # Walk backwards through messages, collecting complete turns
    turns_collected = 0
    relevant = []
    for msg in reversed(messages):
        if msg["role"] in ("user", "assistant"):
            relevant.append(msg)
        if msg["role"] == "user":
            turns_collected += 1
        if turns_collected >= max_turns:
            break
    # Reverse to chronological order
    relevant = list(reversed(relevant))
    for msg in relevant:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant answers to keep the prompt lean
        content = msg["content"]
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines)


def rewrite_query(raw_query, conversation_history):
    """
    Rewrites a follow-up question into a fully self-contained query
    using conversation history for better retrieval.
    """
    if not conversation_history.strip():
        return raw_query

    system_prompt = """You rewrite follow-up medical questions into one standalone search query.
Include necessary context from the conversation (disease, symptoms, topic).
If the follow-up is unrelated to the conversation (sports, coding, trivia, etc.), return it unchanged.
Return only the rewritten question — no explanation, quotes, or preamble."""

    user_prompt = f"""Conversation:
{conversation_history}

Follow-up Question: {raw_query}

Standalone Question:"""

    rewritten = call_groq(system_prompt, user_prompt, max_tokens=120, temperature=0.0)

    if rewritten is None:
        tokenizer, model = get_local_llm()
        rewrite_prompt = f"""Given the conversation below, rewrite the follow-up question as a complete standalone question that includes all necessary context.

Conversation:
{conversation_history}

Follow-up Question: {raw_query}

Standalone Question:"""

        inputs = tokenizer(
            rewrite_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not rewritten or len(rewritten) < 5:
        return raw_query

    return rewritten


# --- UI LAYOUT ---

st.title("⚕️ MediTrust AI")
st.markdown("Your trustworthy, RAG-powered medical assistant. Powered by Hybrid Search and Confidence Scoring.")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    chunks = len(chunked_df) if chunked_df is not None else 0
    st.markdown(f"<div><span class='status-dot'></span><b>Data:</b> {chunks} chunks loaded</div>", unsafe_allow_html=True)
    st.markdown(f"<div><span class='status-dot'></span><b>Embeddings:</b> all-MiniLM-L6-v2</div>", unsafe_allow_html=True)
    st.markdown(f"<div><span class='status-dot'></span><b>Reranker:</b> BAAI/bge-reranker-base</div>", unsafe_allow_html=True)
    if os.getenv("GROQ_API_KEY"):
        st.markdown(f"<div><span class='status-dot'></span><b>LLM:</b> Groq ({GROQ_MODEL})</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div><span class='status-dot-warning'></span><b>LLM:</b> local fallback</div>", unsafe_allow_html=True)
    st.markdown(f"<div><span class='status-dot'></span><b>DB:</b> Qdrant Hybrid Search</div>", unsafe_allow_html=True)
    st.markdown(f"<div><span class='status-dot'></span><b>Memory:</b> Conversational (3 turns)</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. **Hybrid Retrieval**: BM25 (Sparse) + Qdrant (Dense)")
    st.markdown("2. **Cross-Encoder Reranking**: BAAI model ensures relevance.")
    st.markdown("3. **Confidence Scoring**: Aborts if relevance is low or the question is out of scope.")
    st.markdown("4. **Grounded Generation**: Groq LLM answers using only retrieved context.")
    st.markdown("5. **🧠 Query Rewriting**: Rewrites follow-ups using conversation history.")

    # Show live memory in sidebar when conversation is active
    if st.session_state.get("messages"):
        st.markdown("---")
        st.markdown("### 🧠 Active Memory")
        history = build_conversation_history(st.session_state.messages)
        if history:
            st.caption(history)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("View Source & Confidence"):
                details = message["details"]
                
                # Confidence badge
                conf_color = "red" if details['confidence'] == 'Low' else "orange" if details['confidence'] == 'Medium' else "green"
                st.markdown(f"**Confidence:** <span style='color:{conf_color}; font-weight:bold;'>{details['confidence']}</span>", unsafe_allow_html=True)
                
                # Sources
                st.markdown("#### References")
                for idx, src in enumerate(details["sources"]):
                    url = src.get('url', '')
                    url_html = f"<br><a class='source-link' href='{url}' target='_blank'>Read Full Article ↗</a>" if url and str(url).lower() != 'nan' else ""
                    
                    st.markdown(f"""
                        <div class='source-card'>
                            <div class='source-title'>{idx+1}. {src.get('topic', 'Medical Reference')}</div>
                            <div style='color: #475569; font-size: 0.9rem;'>{src.get('question', '')}</div>
                            {url_html}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Explanations
                st.markdown("#### Retrieved Context Preview")
                for idx, exp in enumerate(details["retrieval_explanations"]):
                    st.markdown(f"**Context {idx+1}** (Score: {exp['rerank_score']})")
                    st.info(exp['context_preview'] + "...")

# Accept user input
prompt = st.chat_input("Ask a medical question... (e.g., 'What are symptoms of diabetes?')")

if not st.session_state.messages and not prompt:
    st.markdown("""
        <div style='text-align: center; margin-top: 50px; margin-bottom: 50px;'>
            <h2 style='color: #f8fafc;'>Welcome to MediTrust AI ⚕️</h2>
            <p style='color: #94a3b8; font-size: 1.1rem;'>Your trusted, anti-hallucination medical assistant powered by Hybrid RAG.</p>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("🩺 What are the symptoms of diabetes?", use_container_width=True):
            prompt = "What are the symptoms of diabetes?"
    with cols[1]:
        if st.button("💊 What are the side effects of ibuprofen?", use_container_width=True):
            prompt = "What are the side effects of ibuprofen?"
    with cols[2]:
        if st.button("🔬 How is multiple sclerosis diagnosed?", use_container_width=True):
            prompt = "How is multiple sclerosis diagnosed?"

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate Assistant Response
    with st.chat_message("assistant"):
        # --- QUERY REWRITING ---
        conversation_history = build_conversation_history(
            # Exclude the message we just appended (the current user prompt)
            st.session_state.messages[:-1]
        )
        with st.spinner("🧠 Contextualizing query..."):
            if is_out_of_domain(prompt, conversation_history):
                result = make_out_of_domain_response(prompt)
                search_query = prompt
            else:
                search_query = (
                    rewrite_query(prompt, conversation_history)
                    if conversation_history.strip()
                    else prompt
                )
                result = None

        # Show the rewritten query as a visual indicator when it differs
        if search_query.strip().lower() != prompt.strip().lower():
            st.caption(f"🔍 **Internal search query:** *{search_query}*")

        with st.spinner("Searching and synthesizing..."):
            if result is None:
                result = trustworthy_rag(search_query)
            
            answer = result["answer"]
            st.markdown(answer)
            
            # If we have details to show
            if result["sources"] or result["confidence"]:
                with st.expander("View Source & Confidence"):
                    conf_color = "red" if result['confidence'] == 'Low' else "orange" if result['confidence'] == 'Medium' else "green"
                    st.markdown(f"**Confidence:** <span style='color:{conf_color}; font-weight:bold;'>{result['confidence']}</span>", unsafe_allow_html=True)
                    
                    if result["sources"]:
                        st.markdown("#### References")
                        for idx, src in enumerate(result["sources"]):
                            url = src.get('url', '')
                            url_html = f"<br><a class='source-link' href='{url}' target='_blank'>Read Full Article ↗</a>" if url and str(url).lower() != 'nan' else ""
                            
                            st.markdown(f"""
                                <div class='source-card'>
                                    <div class='source-title'>{idx+1}. {src.get('topic', 'Medical Reference')}</div>
                                    <div style='color: #94a3b8; font-size: 0.9rem;'>{src.get('question', '')}</div>
                                    {url_html}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    if result["retrieval_explanations"]:
                        st.markdown("#### Retrieved Context Preview")
                        for idx, exp in enumerate(result["retrieval_explanations"]):
                            st.markdown(f"**Context {idx+1}** (Score: {exp['rerank_score']})")
                            st.info(exp['context_preview'] + "...")
                
                # Save to history with details
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "details": {
                        "confidence": result["confidence"],
                        "sources": result["sources"],
                        "retrieval_explanations": result["retrieval_explanations"]
                    }
                })
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})
