# --- CELL 0 ---
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# --- CELL 1 ---
import pandas as pd

medical_df = pd.read_csv(
    "/kaggle/input/datasets/summersamar/medical-q-and-a-data/all_questions_answers.csv"
)

print("\nFinal Shape:", medical_df.shape)

medical_df.head()

# --- CELL 2 ---
print(medical_df.columns)

# --- CELL 3 ---
medical_df["context"] = (
    "Topic: " + medical_df["topic"].astype(str) +
    "\nFocus: " + medical_df["focus"].astype(str) +
    "\nQuestion Type: " + medical_df["qtype"].astype(str) +
    "\nQuestion: " + medical_df["question"].astype(str) +
    "\nAnswer: " + medical_df["answer"].astype(str)
)

medical_df[["context"]].head()

# --- CELL 4 ---
medical_df["question_length"] = (
    medical_df["question"]
    .astype(str)
    .apply(len)
)

medical_df["question_length"].describe()

# --- CELL 5 ---
medical_df["answer_length"] = (
    medical_df["answer"]
    .astype(str)
    .apply(len)
)

medical_df["answer_length"].describe()

# --- CELL 6 ---
medical_df["topic"].value_counts()

# --- CELL 7 ---
import matplotlib.pyplot as plt

medical_df["topic"].value_counts().plot(
    kind="bar",
    figsize=(10,5)
)

plt.title("Medical Topics Distribution")
plt.xlabel("Topic")
plt.ylabel("Count")

plt.show()

# --- CELL 8 ---
!pip install scikit-learn

# --- CELL 9 ---
!pip install -q langchain
!pip install -q langchain-text-splitters

# --- CELL 10 ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

documents = []

for idx, row in medical_df.iterrows():

    topic = str(row["topic"])
    focus = str(row["focus"])
    qtype = str(row["qtype"])
    question = str(row["question"])
    answer = str(row["answer"])
    url = str(row["url"])

    full_text = f"""
    Topic: {topic}

    Focus: {focus}
    
    Question Type: {qtype}

    Question:
    {question}

    Answer:
    {answer}
    """

    chunks = text_splitter.split_text(full_text)

    for chunk_id, chunk in enumerate(chunks):

        # Remove tiny/useless chunks
        if len(chunk.strip()) < 150:
            continue

        documents.append({
            "topic": topic,
            "focus": focus,
            "qtype": qtype,
            "question": question,
            "url": url,
            "chunk_id": chunk_id,
            "context": chunk
        })

chunked_df = pd.DataFrame(documents)

print("Final Chunked Shape:", chunked_df.shape)

chunked_df.head()

# --- CELL 11 ---
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(
    chunked_df["context"]
)

print(tfidf_matrix.shape)

# --- CELL 12 ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_tfidf(query, top_k=3):

    # Convert query into TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get top matching chunk indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for idx in top_indices:

        results.append({
            "topic": chunked_df.iloc[idx]["topic"],
            "question": chunked_df.iloc[idx]["question"],
            "chunk_id": chunked_df.iloc[idx]["chunk_id"],
            "context": chunked_df.iloc[idx]["context"],
            "score": similarities[idx]
        })

    return results

# --- CELL 13 ---
query = "What are symptoms of diabetes?"

results = retrieve_tfidf(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("SCORE:", round(r["score"], 4))

    print("\nRETRIEVED CONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 14 ---
!pip install -q sentence-transformers

# --- CELL 15 ---
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# --- CELL 16 ---
embeddings = embedding_model.encode(
    chunked_df["context"].tolist(),
    show_progress_bar=True,
    batch_size=64,
    convert_to_numpy=True
)

# --- CELL 17 ---
embeddings.shape

# --- CELL 18 ---
import numpy as np

np.save("medical_embeddings.npy", embeddings)

# --- CELL 19 ---
chunked_df.to_csv(
    "chunked_medical_dataset.csv",
    index=False
)

# --- CELL 20 ---
embeddings = np.load("/kaggle/input/datasets/aaryan801/embeddings-chunks/medical_embeddings.npy")

print(embeddings.shape)

# --- CELL 21 ---
chunked_df = pd.read_csv("/kaggle/input/datasets/aaryan801/embeddings-chunks/chunked_medical_dataset (1).csv")

chunked_df.head()

# --- CELL 22 ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CELL 23 ---
def semantic_search(query, top_k=5):

    # Convert query into embedding
    query_embedding = embedding_model.encode([query])

    # Compute cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # Get top matching indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for idx in top_indices:

        results.append({
            "topic": chunked_df.iloc[idx]["topic"],
            "question": chunked_df.iloc[idx]["question"],
            "chunk_id": chunked_df.iloc[idx]["chunk_id"],
            "context": chunked_df.iloc[idx]["context"],
            "score": similarities[idx]
        })

    return results

# --- CELL 24 ---
query = "What causes high blood sugar?"

results = semantic_search(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("CHUNK ID:", r["chunk_id"])
    print("SCORE:", round(r["score"], 4))

    print("\nRETRIEVED CONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 25 ---
!pip install -q qdrant-client sentence-transformers tqdm

# --- CELL 26 ---
!pip install -q langchain langchain-community

# --- CELL 27 ---
QDRANT_URL = "https://7ec5a8d5-8bce-4eae-b8ab-c9b7720c4ff9.eu-west-2-0.aws.cloud.qdrant.io"

QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZTBkMWMwNGUtNTQ4Zi00M2YxLTgxMTMtYzA2NmIwZGRjMzkyIn0.UAw-qzOsV8O6OsveY3qr987Xv2MDHYsCkeDKPeXwNq8"

# --- CELL 28 ---
from qdrant_client import QdrantClient

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

print("Connected Successfully!")

# --- CELL 29 ---
from qdrant_client.models import (
    Distance,
    VectorParams
)

client.create_collection(
    collection_name="meditrust_rag_v2",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# --- CELL 30 ---
client.get_collections()

# --- CELL 31 ---
client.get_collection("meditrust_rag_v2")

# --- CELL 32 ---
payloads = []

for idx, row in chunked_df.iterrows():

    payloads.append({
        "topic": row["topic"],
        "focus": row["focus"],
        "qtype": row["qtype"],
        "question": row["question"],
        "url": row["url"],
        "context": row["context"],
        "chunk_id": int(row["chunk_id"])
    })

print("Payload count:", len(payloads))

# --- CELL 33 ---
client.upload_collection(
    collection_name="meditrust_rag_v2",
    vectors=embeddings,
    payload=payloads,
    ids=list(range(len(payloads)))
)

# --- CELL 34 ---
collection_info = client.get_collection(
    "meditrust_rag_v2"
)

print(collection_info)

# --- CELL 35 ---
def qdrant_search(query, top_k=5):

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Query Qdrant
    results = client.query_points(
        collection_name="meditrust_rag_v2",
        query=query_embedding,
        limit=top_k
    )

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

# --- CELL 36 ---
query = "What are symptoms of diabetes?"

results = qdrant_search(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("CHUNK ID:", r["chunk_id"])
    print("SCORE:", round(r["score"], 4))

    print("\nRETRIEVED CONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 37 ---
query = "What causes high blood sugar?"

results = qdrant_search(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("CHUNK ID:", r["chunk_id"])
    print("SCORE:", round(r["score"], 4))

    print("\nRETRIEVED CONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 38 ---
!pip install -q rank-bm25

# --- CELL 39 ---
from rank_bm25 import BM25Okapi

# --- CELL 40 ---
corpus = chunked_df["context"].tolist()

tokenized_corpus = [
    doc.lower().split()
    for doc in corpus
]

# --- CELL 41 ---
bm25 = BM25Okapi(tokenized_corpus)

# --- CELL 42 ---
import numpy as np

def bm25_search(query, top_k=5):

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
            "chunk_id": chunked_df.iloc[idx]["chunk_id"]
        })

    return results

# --- CELL 43 ---
query = "What are symptoms of diabetes?"

results = bm25_search(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("CHUNK ID:", r["chunk_id"])
    print("SCORE:", round(r["score"], 4))

    print("\nCONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 44 ---
from collections import defaultdict
import numpy as np

def hybrid_search(query, top_k=5):

    # -------------------------
    # Dense Retrieval
    # -------------------------
    dense_results = qdrant_search(
        query,
        top_k=10
    )

    # -------------------------
    # BM25 Retrieval
    # -------------------------
    sparse_results = bm25_search(
        query,
        top_k=10
    )

    # -------------------------
    # Normalize Dense Scores
    # -------------------------
    dense_scores = np.array([
        r["score"]
        for r in dense_results
    ])

    dense_scores = (
        dense_scores - dense_scores.min()
    ) / (
        dense_scores.max() - dense_scores.min() + 1e-8
    )

    # -------------------------
    # Normalize BM25 Scores
    # -------------------------
    bm25_scores = np.array([
        r["score"]
        for r in sparse_results
    ])

    bm25_scores = (
        bm25_scores - bm25_scores.min()
    ) / (
        bm25_scores.max() - bm25_scores.min() + 1e-8
    )

    # -------------------------
    # Combine Scores
    # -------------------------
    combined_scores = defaultdict(float)
    combined_docs = {}

    # Dense weighting
    for r, score in zip(
        dense_results,
        dense_scores
    ):

        key = r["context"]

        combined_scores[key] += (
            0.6 * float(score)
        )

        combined_docs[key] = r

    # BM25 weighting
    for r, score in zip(
        sparse_results,
        bm25_scores
    ):

        key = r["context"]

        combined_scores[key] += (
            0.4 * float(score)
        )

        combined_docs[key] = r

    # -------------------------
    # Final Ranking
    # -------------------------
    ranked = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for context, score in ranked[:top_k]:

        doc = combined_docs[context]

        doc["hybrid_score"] = score

        final_results.append(doc)

    return final_results

# --- CELL 45 ---
query = "What are symptoms of diabetes?"

results = hybrid_search(query)

for r in results:

    print("\nTOPIC:", r["topic"])
    print("QUESTION:", r["question"])
    print("CHUNK ID:", r["chunk_id"])
    print("HYBRID SCORE:", round(r["hybrid_score"], 4))

    print("\nCONTEXT:\n")
    print(r["context"][:1000])

    print("\n" + "="*100)

# --- CELL 46 ---
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# --- CELL 47 ---
def rerank_results(
    query,
    retrieved_docs,
    top_k=3
):

    # Build query-document pairs
    pairs = [
        (query, doc["context"])
        for doc in retrieved_docs
    ]

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Attach rerank scores
    for doc, score in zip(
        retrieved_docs,
        scores
    ):
        doc["rerank_score"] = float(score)

    # Sort by rerank score
    reranked = sorted(
        retrieved_docs,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]

# --- CELL 48 ---
query = "What are symptoms of diabetes?"

# Hybrid retrieval
retrieved = hybrid_search(
    query,
    top_k=10
)

# Reranking
final_docs = rerank_results(
    query,
    retrieved,
    top_k=3
)

for doc in final_docs:

    print("\nTOPIC:", doc["topic"])
    print("QUESTION:", doc["question"])
    print("RERANK SCORE:", round(doc["rerank_score"], 4))

    print("\nCONTEXT:\n")
    print(doc["context"][:1000])

    print("\n" + "="*100)

# --- CELL 49 ---
!pip install -q transformers sentencepiece accelerate

# --- CELL 50 ---
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import torch

# --- CELL 51 ---
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# --- CELL 52 ---
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
)

# --- CELL 53 ---
input_text = "Explain diabetes in simple words."

inputs = tokenizer(
    input_text,
    return_tensors="pt"
)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)

# --- CELL 54 ---
def build_context(docs):

    contexts = []

    for doc in docs:

        contexts.append(
            f"""
            Topic: {doc.get('topic', 'Unknown')}
            Focus: {doc.get('focus', 'Unknown')}
            Question Type: {doc.get('qtype', 'Unknown')}

            {doc.get('context', '')}
            """
        )

    return "\n\n".join(contexts)

# --- CELL 55 ---
def build_prompt(query, context):

    prompt = f"""
You are a helpful medical assistant.

Answer the user's question ONLY using the provided medical context.

If the answer is not present in the context, say:
"I could not find reliable medical information in the retrieved documents."

Medical Context:
{context}

Question:
{query}

Answer:
"""

    return prompt

# --- CELL 56 ---
def generate_rag_answer(
    query,
    retrieval_k=10,
    final_k=3
):

    # ---------------------------------
    # Step 1 — Hybrid Retrieval
    # ---------------------------------
    retrieved_docs = hybrid_search(
        query,
        top_k=retrieval_k
    )

    # ---------------------------------
    # Step 2 — Reranking
    # ---------------------------------
    reranked_docs = rerank_results(
        query,
        retrieved_docs,
        top_k=final_k
    )

    print("\nDEBUG")
    print(type(reranked_docs))
    print(len(reranked_docs))

    for i, doc in enumerate(reranked_docs[:3]):
        print(f"\nDOC {i+1}")
        print(doc.keys())
    # ---------------------------------
    # Step 3 — Build Context
    # ---------------------------------
    context = build_context(
        reranked_docs
    )

    # ---------------------------------
    # Step 4 — Build Prompt
    # ---------------------------------
    prompt = build_prompt(
        query,
        context
    )

    # ---------------------------------
    # Step 5 — Tokenize
    # ---------------------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    # ---------------------------------
    # Step 6 — Generate
    # ---------------------------------
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3
    )

    # ---------------------------------
    # Step 7 — Decode
    # ---------------------------------
    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": reranked_docs
    }

# --- CELL 57 ---
result = generate_rag_answer(
    "What are symptoms of diabetes?"
)

print("\nQUESTION:\n")
print(result["query"])

print("\nANSWER:\n")
print(result["answer"])

# --- CELL 58 ---
def extract_sources(retrieved_docs):

    sources = []

    seen = set()

    for doc in retrieved_docs:

        source = {
            "topic": doc["topic"],
            "focus": doc["focus"],
            "question": doc["question"],
            "url": doc["url"]
        }

        key = doc["url"]

        if key not in seen:

            sources.append(source)
            seen.add(key)

    return sources

# --- CELL 59 ---
def compute_confidence(retrieved_docs):

    avg_score = np.mean([
        doc["rerank_score"]
        for doc in retrieved_docs
    ])

    if avg_score > 0.95:
        return "High"

    elif avg_score > 0.85:
        return "Medium"

    else:
        return "Low"

# --- CELL 60 ---
def trustworthy_rag(query):

    # ---------------------------------
    # Generate RAG Answer
    # ---------------------------------
    result = generate_rag_answer(query)

    # ---------------------------------
    # Extract Sources
    # ---------------------------------
    sources = extract_sources(
        result["retrieved_docs"]
    )

    # ---------------------------------
    # Compute Confidence
    # ---------------------------------
    confidence = compute_confidence(
        result["retrieved_docs"]
    )

    return {
        "query": result["query"],
        "answer": result["answer"],
        "sources": sources,
        "confidence": confidence
    }

# --- CELL 61 ---
result = trustworthy_rag(
    "What are symptoms of diabetes?"
)

print("\nQUESTION:\n")
print(result["query"])

print("\nANSWER:\n")
print(result["answer"])

print("\nCONFIDENCE:\n")
print(result["confidence"])

print("\nSOURCES:\n")

for idx, source in enumerate(
    result["sources"]
):

    print(
        f"{idx+1}. "
        f"{source['topic']} "
        f"→ "
        f"{source['question']}"
    )

# --- CELL 62 ---
def is_low_confidence(retrieved_docs):

    avg_score = np.mean([
        doc["rerank_score"]
        for doc in retrieved_docs
    ])

    return avg_score < 0.80

# --- CELL 63 ---
def trustworthy_rag(query):

    # ---------------------------------
    # Generate RAG Answer
    # ---------------------------------
    result = generate_rag_answer(query)

    retrieved_docs = result["retrieved_docs"]

    # ---------------------------------
    # Low Confidence Detection
    # ---------------------------------
    if is_low_confidence(retrieved_docs):

        return {
            "query": query,
            "answer":
                "I could not find reliable "
                "medical information in the "
                "retrieved documents.",
            "confidence": "Low",
            "sources": []
        }

    # ---------------------------------
    # Extract Sources
    # ---------------------------------
    sources = extract_sources(
        retrieved_docs
    )

    # ---------------------------------
    # Compute Confidence
    # ---------------------------------
    confidence = compute_confidence(
        retrieved_docs
    )

    return {
        "query": result["query"],
        "answer": result["answer"],
        "sources": sources,
        "confidence": confidence
    }

# --- CELL 64 ---
result = trustworthy_rag(
    "What are symptoms of diabetes?"
)

print(result)

# --- CELL 65 ---
result = trustworthy_rag(
    "Can diabetes cure brain cancer naturally?"
)

print(result)

# --- CELL 66 ---
def explain_retrieval(retrieved_docs):

    explanations = []

    for doc in retrieved_docs:

        explanations.append({

            "topic": doc["topic"],

            "question": doc["question"],

            "rerank_score": round(
                doc["rerank_score"],
                4
            ),

            "context_preview":
                doc["context"][:300]
        })

    return explanations

# --- CELL 67 ---
def trustworthy_rag(query):

    # ---------------------------------
    # Generate RAG Answer
    # ---------------------------------
    result = generate_rag_answer(query)

    retrieved_docs = result["retrieved_docs"]

    # ---------------------------------
    # Low Confidence Detection
    # ---------------------------------
    if is_low_confidence(retrieved_docs):

        return {
            "query": query,
            "answer":
                "I could not find reliable "
                "medical information in the "
                "retrieved documents.",
            "confidence": "Low",
            "sources": [],
            "retrieval_explanations": []
        }

    # ---------------------------------
    # Extract Sources
    # ---------------------------------
    sources = extract_sources(
        retrieved_docs
    )

    # ---------------------------------
    # Compute Confidence
    # ---------------------------------
    confidence = compute_confidence(
        retrieved_docs
    )

    # ---------------------------------
    # Retrieval Transparency
    # ---------------------------------
    retrieval_explanations = (
        explain_retrieval(
            retrieved_docs
        )
    )

    return {

        "query": result["query"],

        "answer": result["answer"],

        "sources": sources,

        "confidence": confidence,

        "retrieval_explanations":
            retrieval_explanations
    }

# --- CELL 68 ---
result = trustworthy_rag(
    "Who won IPL 2025?"
)

print("\nQUESTION:\n")
print(result["query"])

print("\nANSWER:\n")
print(result["answer"])

print("\nCONFIDENCE:\n")
print(result["confidence"])

print("\nSOURCES:\n")

for idx, source in enumerate(
    result["sources"]
):

    print(
        f"{idx+1}. "
        f"{source['topic']} "
        f"→ "
        f"{source['question']}"
    )

    print(
        f"   URL: {source['url']}"
    )

print("\nRETRIEVAL EXPLANATIONS:\n")

for exp in result[
    "retrieval_explanations"
]:

    print("\nTOPIC:", exp["topic"])

    print(
        "QUESTION:",
        exp["question"]
    )

    print(
        "RERANK SCORE:",
        exp["rerank_score"]
    )

    print(
        "\nCONTEXT PREVIEW:\n",
        exp["context_preview"]
    )

    print("\n" + "="*100)

# --- CELL 69 ---
for doc in retrieved_docs:
    print(doc.get("url"))

# --- CELL 70 ---
result = generate_rag_answer(
    "What are symptoms of diabetes?"
)

for i, doc in enumerate(result["retrieved_docs"]):
    print(f"\nDOC {i+1}")
    print("QUESTION:", doc.get("question"))
    print("URL:", doc.get("url"))

# --- CELL 71 ---
print(result["sources"])

# --- CELL 72 ---
def test_out_of_domain():

    queries = [

        "Who won IPL 2025?",
        "What is the capital of France?",
        "Write a Python binary search program.",
        "Who is the CEO of Google?",
        "Explain quantum computing.",
        "Best laptop under 1000 dollars?",
        "How to make chocolate cake?",
        "Who won the FIFA World Cup?",
        "What is the weather in New York?",
        "Explain operating system scheduling."
    ]

    print("=" * 100)
    print("OUT-OF-DOMAIN TESTING")
    print("=" * 100)

    for query in queries:

        result = trustworthy_rag(query)

        print("\nQUERY:")
        print(query)

        print("\nCONFIDENCE:")
        print(result["confidence"])

        print("\nANSWER:")
        print(result["answer"])

        print("\n" + "-" * 100)

# --- CELL 73 ---
query = "Who won IPL 2025?"

retrieved = hybrid_search(query)

reranked = rerank_documents(
    query,
    retrieved
)

print(type(reranked))
print(len(reranked))

for i, doc in enumerate(reranked[:3]):
    print(f"\nDOC {i+1}")
    print(doc.keys())

# --- CELL 74 ---
test_out_of_domain()

# --- CELL 75 ---
print(generate_rag_answer)

# --- CELL 76 ---
trustworthy_rag("Who won IPL 2025?")

# --- CELL 77 ---
query = "Who won IPL 2025?"

retrieved = hybrid_search(query)

for i, doc in enumerate(retrieved):

    print(f"\nDOC {i+1}")
    print(doc.keys())

    if "focus" not in doc:
        print("\nFOUND OLD DOCUMENT")
        print(doc)

# --- CELL 78 ---
print(client.get_collection(
    collection_name="meditrust_rag_v2"
))

# --- CELL 79 ---
results = client.query_points(
    collection_name="meditrust_rag_v2",
    limit=1
)

# --- CELL 80 ---
trustworthy_rag("Does garlic cure cancer?")

# --- CELL 81 ---
trustworthy_rag("Can vaccines cause autism?")

# --- CELL 82 ---
bm25_documents[0].keys()

# --- CELL 83 ---
chunks_df.columns

# --- CELL 84 ---


