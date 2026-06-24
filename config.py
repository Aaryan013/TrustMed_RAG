import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
REFUSAL_ANSWER = "I could not find reliable medical information in the retrieved documents."
QUERY_ALIGNMENT_THRESHOLD = 0.55
