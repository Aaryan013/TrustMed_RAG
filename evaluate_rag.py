import sys
from unittest.mock import MagicMock
import time
import pandas as pd

# We mock Streamlit to prevent the UI components from rendering when we import app.py
class MockStreamlit:
    def __init__(self):
        self.session_state = MagicMock()
        self.session_state.messages = []
        
    def chat_input(self, *args, **kwargs):
        return None
        
    def __getattr__(self, name):
        # Allow decorators to pass through the functions unchanged
        if name in ['cache_resource', 'cache_data']:
            return lambda func=None, **kwargs: (lambda f: f) if func is None else func
        # Mock other Streamlit functions
        return MagicMock()

sys.modules['streamlit'] = MockStreamlit()

import app  # Import your actual functions!

def run_evaluation():
    questions = [
        {"Category": "Easy", "Query": "What are the common symptoms of diabetes?"},
        {"Category": "Easy", "Query": "What causes high blood pressure?"},
        
        {"Category": "Moderate", "Query": "How is pneumonia typically treated?"},
        {"Category": "Moderate", "Query": "What are the common side effects of taking ibuprofen?"},
        
        {"Category": "Hard", "Query": "What are the diagnostic criteria for multiple sclerosis?"},
        {"Category": "Hard", "Query": "How does metformin affect blood glucose levels at a cellular level?"},
        
        {"Category": "Tricky (Out of Domain)", "Query": "Who won the World Cup in 2022?"},
        {"Category": "Tricky (Out of Domain)", "Query": "Can you write a python script to sort an array?"},
        
        {"Category": "Tricky (No Context/Fictional)", "Query": "What is the recommended treatment for chronotemporal displacement syndrome?"}
    ]

    results = []
    print("="*60)
    print("Starting Automated RAG Evaluation...")
    print("="*60 + "\n")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Category: {q['Category']}")
        print(f"Query: {q['Query']}")
        
        start_time = time.time()
        
        # Test the routing logic
        if app.is_out_of_domain(q['Query']):
            res = app.make_out_of_domain_response(q['Query'])
        else:
            res = app.trustworthy_rag(q['Query'])
            
        latency = time.time() - start_time
        
        # Collect results
        answer = res.get("answer", "")
        confidence = res.get("confidence", "N/A")
        sources = len(res.get("sources", []))
        
        results.append({
            "Category": q['Category'],
            "Query": q['Query'],
            "Answer": answer,
            "Confidence": confidence,
            "Latency (s)": round(latency, 2),
            "Sources Count": sources
        })
        
        # Truncate answer for clean CLI output
        display_answer = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"Confidence: {confidence} | Latency: {round(latency, 2)}s | Sources: {sources}")
        print(f"Answer: {display_answer}\n")
        print("-" * 60)

    # Save Results to CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    
    print("\n✅ Evaluation complete!")
    print("Results have been saved to 'evaluation_results.csv'.")
    print("You can open this file in Excel to review the RAG system's performance.")

if __name__ == "__main__":
    run_evaluation()
