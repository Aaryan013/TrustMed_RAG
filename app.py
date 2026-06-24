import streamlit as st
import os

# --- UI LAYER & IMPORTS ---
from config import GROQ_MODEL
from src.ui import inject_custom_css, clean_thought_tags, update_sidebar_library
from src.models import load_models, connect_qdrant
from src.retrieval import load_data_and_bm25
from src.rag import (
    trustworthy_rag,
    is_out_of_domain,
    make_out_of_domain_response,
    build_conversation_history,
    rewrite_query
)

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="MediTrust AI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# Initialize components
with st.spinner("Initializing models... (This may take a minute on first run)"):
    embedding_model, reranker = load_models()
    qdrant_client = connect_qdrant()
    chunked_df, bm25 = load_data_and_bm25()

if chunked_df is None:
    st.error("Data not found. Please run `python data_prep.py` first to generate the chunked dataset.")
    st.stop()

# --- UI LAYOUT ---

st.title("⚕️ MediTrust AI")
st.markdown("Your trustworthy, RAG-powered medical assistant. Powered by Hybrid Search and Confidence Scoring.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# Refresh sidebar data tracking variables to reflect new turns
update_sidebar_library(st.session_state.messages)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(clean_thought_tags(message["content"]))
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
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        conversation_history = build_conversation_history(
            st.session_state.messages[:-1]
        )
        with st.spinner("🧠 Contextualizing query..."):
            if is_out_of_domain(prompt, embedding_model, conversation_history):
                result = make_out_of_domain_response(prompt)
                search_query = prompt
            else:
                search_query = (
                    rewrite_query(prompt, conversation_history)
                    if conversation_history.strip()
                    else prompt
                )
                result = None

        if search_query.strip().lower() != prompt.strip().lower():
            st.caption(f"🔍 **Internal search query:** *{search_query}*")

        with st.spinner("Searching and synthesizing..."):
            if result is None:
                result = trustworthy_rag(search_query, qdrant_client, embedding_model, reranker, chunked_df, bm25)
            
            answer = result["answer"]
            st.markdown(clean_thought_tags(answer))
            
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
