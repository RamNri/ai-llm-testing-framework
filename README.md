# AI / LLM Testing Framework

A practical AI/LLM testing framework built incrementally with Python, Pytest, DeepEval, OpenAI, and FAISS.

The goal is to explore and implement a production-oriented testing approach for modern AI systems, including RAG applications, LLM applications, AI agents, conversational systems, safety, traditional ML validation, performance, and CI/CD quality gates.

---

## Current Status

### RAG Retrieval Evaluation — Implemented

The first implemented capability is RAG retrieval evaluation using a real Samsung Galaxy S22 user manual.

Current pipeline:

```text
PDF Document
     ↓
Text Extraction
     ↓
Recursive Text Chunking
     ↓
OpenAI Embeddings
     ↓
FAISS Vector Store
     ↓
Top-K Retrieval
     ↓
DeepEval Evaluation

#Current implementation
	• PDF text extraction using PyMuPDF 
	• Recursive text chunking using LangChain 
	• OpenAI text-embedding-3-small embeddings 
	• FAISS vector store 
	• Persistent FAISS index 
	• Separate index-building and test-time loading 
	• Top-K similarity retrieval 
	• DeepEval Contextual Precision 
	• DeepEval Contextual Recall 
	• Positive and negative retrieval experiments 
	• Golden/reference answer based on source documentation 
	• SQLite-backed DeepEval local evaluation storage 

#Current retrieval baseline
For the initial Dark Mode test case:
Metric	                  Result
Document chunks	          511
Top-K	                    3
Contextual Precision	    1.00
Contextual Recall       	1.00
These scores represent the current test case and are not intended to represent overall framework accuracy.

#RAG Testing

The RAG testing approach separates retrieval quality from generation quality.

Retrieval
User Query
    ↓
Embedding
    ↓
FAISS Similarity Search
    ↓
Top-K Retrieved Context
    ↓
Retrieval Evaluation

Current retrieval metrics:

Contextual Precision

Evaluates whether the retrieved contexts are relevant to the question and appropriately ranked.

Contextual Recall

Evaluates whether the retrieved contexts contain enough information to support the expected/reference answer.

#Project Structure
ai-llm-testing-framework/
│
├── simple_rag.py
├── build_faiss_index.py
│
├── test_rag_retrieval.py
├── test_rag_bad_retrieval.py
├── test_contextual_recall.py
├── test_contextual_recall_bad.py
├── test_contextual_recall_partial.py
├── test_rag_contextual_recall.py
│
├── test_openai.py
├── test_openai_embedding.py
│
├── requirements.txt
├── README.md
└── .gitignore

#Purpose

This project is being developed as a practical portfolio demonstrating AI/LLM quality engineering rather than a collection of isolated AI testing examples.

The long-term goal is to build a reusable testing framework capable of evaluating:

LLM Applications
      │
      ├── RAG
      ├── AI Agents
      ├── Chatbots
      ├── APIs
      └── Safety
            +
Traditional ML Models
            +
Performance / Reliability
            +
Automated Regression
            +
CI/CD Quality Gates