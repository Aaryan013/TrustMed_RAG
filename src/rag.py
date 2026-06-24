import numpy as np
from src.models import call_groq, get_local_llm
from src.retrieval import hybrid_search, rerank_results
from config import REFUSAL_ANSWER, QUERY_ALIGNMENT_THRESHOLD

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

def is_out_of_domain(query, embedding_model, conversation_history=""):
    """Detect non-medical questions before retrieval."""
    system_prompt = """You classify questions for a medical Q&A assistant.
Reply with exactly one word: MEDICAL or NON_MEDICAL.

Use the conversation history when the question is a follow-up (e.g. "What are its types?").
MEDICAL includes: symptoms, diseases, treatments, medications, anatomy, mental health, nutrition for health, public health, clinical tests.
NON_MEDICAL includes: sports, politics, entertainment, programming, shopping, recipes (non-health), weather, general trivia, business, travel."""

    user_prompt = query
    if conversation_history.strip():
        user_prompt = f"Conversation:\n{conversation_history}\n\nQuestion: {query}"

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

def trustworthy_rag(query, qdrant_client, embedding_model, reranker, chunked_df, bm25):
    retrieved_docs = hybrid_search(query, qdrant_client, embedding_model, chunked_df, bm25, top_k=10)
    reranked_docs = rerank_results(query, retrieved_docs, reranker, top_k=3)

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

    user_prompt = f"Medical Context:\n{context}\n\nQuestion:\n{query}"

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

def build_conversation_history(messages, max_turns=3):
    history_lines = []
    turns_collected = 0
    relevant = []
    for msg in reversed(messages):
        if msg["role"] in ("user", "assistant"):
            relevant.append(msg)
        if msg["role"] == "user":
            turns_collected += 1
        if turns_collected >= max_turns:
            break
    relevant = list(reversed(relevant))
    for msg in relevant:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines)

def rewrite_query(raw_query, conversation_history):
    if not conversation_history.strip():
        return raw_query

    system_prompt = """You rewrite follow-up medical questions into one standalone search query.
Include necessary context from the conversation (disease, symptoms, topic).
If the follow-up is unrelated to the conversation (sports, coding, trivia, etc.), return it unchanged.
Return only the rewritten question — no explanation, quotes, or preamble."""

    user_prompt = f"Conversation:\n{conversation_history}\n\nFollow-up Question: {raw_query}\n\nStandalone Question:"

    rewritten = call_groq(system_prompt, user_prompt, max_tokens=120, temperature=0.0)

    if rewritten is None:
        tokenizer, model = get_local_llm()
        rewrite_prompt = f"""Given the conversation below, rewrite the follow-up question as a complete standalone question that includes all necessary context.

Conversation:
{conversation_history}

Follow-up Question: {raw_query}

Standalone Question:"""

        inputs = tokenizer(rewrite_prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not rewritten or len(rewritten) < 5:
        return raw_query

    return rewritten
