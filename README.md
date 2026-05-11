# SmartPDF_chatbot

## Overview
This project is a Retrieval-Augmented Generation (RAG) based AI chatbot that can answer questions from PDF documents using semantic search and Large Language Models.

The chatbot:
- Loads PDF documents
- Splits text into chunks
- Converts chunks into embeddings
- Stores embeddings in ChromaDB
- Retrieves relevant chunks
- Sends context to Gemini AI
- Generates grounded answers

## Technologies Used
- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Google Gemini API
- RAG Architecture

## Features
- Semantic Search
- Vector Database Retrieval
- PDF Question Answering
- Continuous Chat Loop
- Hallucination Reduction
- Context-Aware Responses

## Workflow
PDF → Chunking → Embeddings → Vector DB → Retrieval → Gemini Response

## Future Improvements
- Multi-PDF Support
- Streamlit UI
- Chat History Memory
- Source Citations
- Hybrid Search

## How to Run

### Install dependencies
pip install -r requirements.txt

### Run chatbot
python app.py
