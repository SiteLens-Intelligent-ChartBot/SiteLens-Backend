# College Chatbot

A simple chatbot API that answers questions based on a text file (`college_data.txt`) using semantic search.

## Setup

1. Clone the repo or copy the files into a folder.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows

   uvicorn api:app --reload
