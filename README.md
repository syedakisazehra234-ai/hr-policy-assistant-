# 👩‍💼 HR Policy Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload an HR Policy PDF and ask questions about the policy.

The application retrieves relevant sections from the uploaded document and uses Groq's `openai/gpt-oss-20b` model to generate an answer.

## 🚀 Features

- Upload HR Policy PDF
- Extract PDF text using PyMuPDF
- Split policy text into overlapping chunks
- Generate semantic embeddings using Sentence Transformers
- Store embeddings using FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Display source pages
- Chat-style interface
- Runs entirely through Streamlit
- No database required

## 🧠 RAG Architecture

```text
                    HR Policy PDF
                          |
                          v
                    PyMuPDF
                          |
                          v
                    Text Extraction
                          |
                          v
                     Chunking
                          |
                          v
              Sentence Transformers
                          |
                          v
                    Embeddings
                          |
                          v
                       FAISS
                          |
                          |
User Question ------------+
        |
        v
Question Embedding
        |
        v
FAISS Similarity Search
        |
        v
Relevant Policy Chunks
        |
        v
Groq openai/gpt-oss-20b
        |
        v
       Answer
