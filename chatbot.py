import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pymongo import MongoClient

# -----------------------------
# MongoDB Atlas Setup
# -----------------------------
MONGO_URI = "mongodb+srv://kumarswarup7272_db_user:6SS630zpcEU852EV@cluster0.7x8rukv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "sitelense_chats"
COLLECTION_NAME = "sitelense_ai"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------
# Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


def answer_question(query: str) -> str:
    """Find best matching statement from MongoDB (latest overrides older ones)."""
    docs = list(
        collection.find({}, {"_id": 1, "text": 1, "embedding": 1})
        .sort([("_id", -1)])  # newest → oldest
    )

    if not docs:
        return "No knowledge available yet. Please add statements first."

    query_embedding = model.encode([query])

    for doc in docs:  # check latest first
        if "embedding" not in doc:
            continue

        emb = np.array(doc["embedding"]).reshape(1, -1)
        sim = cosine_similarity(query_embedding, emb)[0][0]

        if sim >= 0.4:  # similarity threshold
            return doc["text"]

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
