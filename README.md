🩺 Retrieval-Augmented Medical Question Answering Assistant
An AI-powered medical question-answering web application that uses Retrieval-Augmented Generation (RAG) to retrieve relevant medical information from trusted sources before generating a response.

📌 Overview
The system allows users to ask medical questions through a simple web interface. It retrieves relevant information from the MedQuAD dataset using semantic search and provides the retrieved context to a Large Language Model (LLM) to generate a clear and context-aware response.

The system is designed to reduce unsupported AI responses and provide source references along with generated answers.

🎯 Objectives
Develop a user-friendly medical question-answering assistant.
Integrate the MedQuAD dataset as the medical knowledge base.
Implement BioBERT embeddings for semantic understanding.
Construct a FAISS vector database for efficient retrieval.
Orchestrate the RAG pipeline using LangChain.
Generate context-aware responses using an LLM.
Provide source references and medical disclaimers.
🏗️ System Workflow
User Login/Register
        ↓
Dashboard
        ↓
Medical Query
        ↓
Query Preprocessing
        ↓
BioBERT Embeddings
        ↓
FAISS Semantic Search
        ↓
Retrieve Relevant MedQuAD Records
        ↓
LangChain Prompt Construction
        ↓
Large Language Model
        ↓
Generate Medical Response
        ↓
Source References + Search History
        ↓
Display Result
