import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient

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

from google import genai
from google.genai import errors
import time, random

client = genai.Client(api_key="AIzaSyDpo-BntSkQLg4jMMCzr09tX8c4I3IHEMc")

def safe_rewrite(original_text: str) -> str:
    """Call Gemini API with retries. Falls back to original_text if it keeps failing."""
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Please answer the user’s question naturally using this information:\n\n{original_text}"
            )
            return response.text if response.text else original_text.split("Context: ", 1)[1]
        
        
        except errors.ClientError as e:  
            if "RESOURCE_EXHAUSTED" in str(e):
                print("❌ Gemini quota exceeded → using fallback.")
                return original_text.split("Context: ", 1)[1]
            raise 


        except errors.ServerError:
            print(f"⚠️ Gemini overloaded (attempt {attempt+1}) → retrying...")
            time.sleep(2 ** attempt + random.random())

    return original_text.split("Context: ", 1)[1]  # fallback






def answer_question(query: str) -> str:
    """Find the best matching statement from MongoDB (highest similarity)."""
    docs = list(
        collection.find({}, {"_id": 1, "text": 1, "embedding": 1})
    )

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
    # Find latest document with this text
    doc = collection.find_one({"text": text}, sort=[("_id", -1)])
    if doc:
        collection.delete_one({"_id": doc["_id"]})
        return True
    return False


def list_statements():
    """Return all statements sorted from latest → oldest"""
    docs = collection.find({}, {"_id": 0, "text": 1}).sort([("_id", -1)])
    return [doc["text"] for doc in docs]
