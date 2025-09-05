import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# File where data is stored
DATA_FILE = "college_data.txt"

# In-memory store
passages = []
passage_embeddings = []

def load_initial_data():
    global passages, passage_embeddings
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            passages[:] = [line.strip() for line in f if line.strip()]
        passage_embeddings[:] = model.encode(passages)
    else:
        passages.clear()
        passage_embeddings.clear()

# Call at startup
load_initial_data()

def save_to_file():
    """Write all passages back to file."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for line in passages:
            f.write(line + "\n")

def answer_question(query: str) -> str:
    if not passages:
        return "No knowledge available yet. Please add statements first."

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < 0.4:
        return "I don't know the answer to that."
    return passages[best_idx]

def add_statement(text: str):
    """Add a new statement dynamically and save."""
    global passages, passage_embeddings
    passages.append(text)
    emb = model.encode([text])[0]
    passage_embeddings.append(emb)
    save_to_file()

def delete_statement(text: str):
    """Delete a statement dynamically and save."""
    global passages, passage_embeddings
    if text in passages:
        idx = passages.index(text)
        passages.pop(idx)
        passage_embeddings.pop(idx)
        save_to_file()
        return True
    return False

def list_statements():
    return passages
