from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import answer_question, add_statement, delete_statement, list_statements
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Question(BaseModel):
    query: str

class Statement(BaseModel):
    text: str

# --- User asks ---
@app.post("/ask")
def ask_question(question: Question):
    answer = answer_question(question.query)
    return {"answer": answer}

# --- Admin adds ---
@app.post("/admin/add")
def admin_add(statement: Statement):
    add_statement(statement.text)
    return {"ok": True, "count": len(list_statements())}

# --- Admin deletes ---
@app.delete("/admin/delete")
def admin_delete(statement: Statement):
    success = delete_statement(statement.text)
    return {"ok": success, "count": len(list_statements())}

# --- Admin lists ---
@app.get("/admin/list")
def admin_list():
    return {"count": len(list_statements()), "statements": list_statements()}

# Run - 
# uvicorn api:app --port 8080 --reload