import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient
import time, random

# -----------------------------
# MongoDB Atlas Setup
# -----------------------------
MONGO_URI = "mongodb+srv://sunilsahoo:2664@cluster0.yp0utdu.mongodb.net"
DB_NAME = "sitelense_chats"
COLLECTION_NAME = "sitelense_ai"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------
# Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Gemini Setup (correct SDK)
# -----------------------------
import google.generativeai as genai

genai.configure(api_key="AIzaSyAPmQlWLJz3XH2PcuoGujeN1okniir6DTU")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def safe_rewrite(original_text: str) -> str:
    """Call Gemini API with retries. Falls back to raw context if it fails."""
    for attempt in range(5):
        try:
            response = gemini_model.generate_content(
                f"Please answer the user’s question naturally using this information:\n\n{original_text}"
            )
            if response and response.text:
                return response.text
            else:
                return original_text.split("Context: ", 1)[1]

        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                print("❌ Gemini quota exceeded → using fallback.")
                return original_text.split("Context: ", 1)[1]

            print(f"⚠️ Gemini error on attempt {attempt+1}: {err}")
            time.sleep(2 ** attempt + random.random())  # exponential backoff

    # Final fallback
    return original_text.split("Context: ", 1)[1]


def answer_question(query: str) -> str:
    """Find the best matching statement from MongoDB (highest similarity)."""
    docs = list(collection.find({}, {"_id": 1, "text": 1, "embedding": 1}))

    if not docs:
        return "No knowledge available yet. Please add statements first."

    query_embedding = model.encode([query]).reshape(1, -1)

    best_doc = None
    best_score = -1

    for doc in docs:
        if "embedding" not in doc:
            continue

        emb = np.array(doc["embedding"]).reshape(1, -1)
        sim = cosine_similarity(query_embedding, emb)[0][0]

        if sim > best_score:  # keep the most similar
            best_score = sim
            best_doc = doc

    if best_doc and best_score >= 0.4:  # threshold check
        return safe_rewrite(f"Q: {query}\nContext: {best_doc['text']}")

    return "I don't know the answer to that."


def add_statement(text: str):
    """Add a new statement to MongoDB with embedding"""
    emb = model.encode([text])[0].tolist()  # convert to list for MongoDB storage
    collection.insert_one({"text": text, "embedding": emb})


def delete_statement(text: str):
    """Delete the latest occurrence of a statement from MongoDB"""
    doc = collection.find_one({"text": text}, sort=[("_id", -1)])
    if doc:
        collection.delete_one({"_id": doc["_id"]})
        return True
    return False


def list_statements():
    """Return all statements sorted from latest → oldest"""
    docs = collection.find({}, {"_id": 0, "text": 1}).sort([("_id", -1)])
    return [doc["text"] for doc in docs]
