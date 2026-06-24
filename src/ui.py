import streamlit as st
import re

def inject_custom_css():
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

def clean_thought_tags(text: str) -> str:
    """Removes the <think>...</think> reasoning blocks dynamically."""
    if not text: return ""
    return re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL).strip()

def update_sidebar_library(messages):
    """Extracts all validated citations across the chat session to populate the Sidebar with full URLs."""
    with st.sidebar:
        st.markdown("### 📚 Reference Library")
        st.markdown("---")
        st.write("Sources used during this active consulting session:")

        global_urls = set()

        for msg in messages:
            if "details" in msg and msg["details"].get("sources"):
                for src in msg["details"]["sources"]:
                    url = src.get('url')
                    if url and str(url).lower() != 'nan':
                        global_urls.add(url)

        if global_urls:
            for url in sorted(global_urls):
                st.markdown(f"• [{url}]({url})")
        else:
            st.info("No documents referenced yet. Query the system to source medical text.")
