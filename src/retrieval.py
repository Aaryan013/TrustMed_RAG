import streamlit as st
import pandas as pd
import numpy as np
import time
import concurrent.futures
from collections import defaultdict
from rank_bm25 import BM25Okapi

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

def qdrant_search(query, qdrant_client, embedding_model, top_k=10, retries=3):
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

def bm25_search(query, chunked_df, bm25, top_k=10):
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

def hybrid_search(query, qdrant_client, embedding_model, chunked_df, bm25, top_k=10):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dense = executor.submit(qdrant_search, query, qdrant_client, embedding_model, top_k)
        future_sparse = executor.submit(bm25_search, query, chunked_df, bm25, top_k)
        
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

def rerank_results(query, retrieved_docs, reranker, top_k=3):
    pairs = [(query, str(doc["context"])) for doc in retrieved_docs]
    if not pairs:
        return []
        
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
        
    reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
