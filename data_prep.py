import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def prepare_data(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    try:
        medical_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}")
        return

    print("Data loaded. Shape:", medical_df.shape)

    # Initialize text splitter
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
    print("Chunking data...")
    
    for idx, row in medical_df.iterrows():
        topic = str(row.get("topic", ""))
        focus = str(row.get("focus", ""))
        qtype = str(row.get("qtype", ""))
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        url = str(row.get("url", ""))

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
    print("Chunking complete. Final Chunked Shape:", chunked_df.shape)

    print(f"Saving chunked data to {output_csv}...")
    chunked_df.to_csv(output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = "all_questions_answers.csv"
    output_file = "chunked_medical_dataset.csv"
    
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Delete it if you want to regenerate.")
    else:
        prepare_data(input_file, output_file)
