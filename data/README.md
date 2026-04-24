# 🧠 RAG-Based Customer Support Assistant (LangGraph + HITL)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG) based Customer Support Assistant** using **LangGraph** and **ChromaDB**.

The system processes a PDF knowledge base, retrieves relevant information using embeddings, and answers user queries. It also includes **intent-based routing** and a **Human-in-the-Loop (HITL)** mechanism for handling sensitive or low-confidence queries.

---

## 🚀 Features

* 📄 PDF-based knowledge ingestion
* 🔍 Semantic search using embeddings (HuggingFace)
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔁 Graph-based workflow using LangGraph
* ⚠️ Intent detection (Knowledge vs Sensitive queries)
* 👨‍💼 Human-in-the-Loop (HITL) escalation
* 📊 Confidence-based response routing

---

## 🏗️ System Architecture

```plaintext
User Query
   ↓
LangGraph Workflow
   ↓
[Process Node]
   ↓
Retrieve Relevant Chunks (ChromaDB)
   ↓
Generate Answer / Compute Confidence
   ↓
Conditional Routing
   ├── High Confidence → Output
   └── Low Confidence → HITL (Human Support)
```

---

## 📂 Project Structure

```plaintext
rag_support_project/
│
├── app/
│   ├── main.py          # LangGraph workflow & chatbot loop
│   ├── ingest.py        # PDF ingestion pipeline
│   ├── state.py         # Graph state definition
│   ├── hitl.py          # Human escalation logic
│   └── config.py        # Configuration settings
│
├── data/
│   └── knowledge_base.pdf   # Input knowledge base
│
├── chroma_store/        # Vector database (auto-generated)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <your-repo-link>
cd rag_support_project
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install langgraph langchain langchain-community chromadb sentence-transformers pypdf langchain-text-splitters
```

---

## 📄 Setup

Place your PDF inside:

```plaintext
data/knowledge_base.pdf
```

---

## 🔄 Run Ingestion

Convert PDF into embeddings and store in ChromaDB:

```bash
python -m app.ingest
```

---

## ▶️ Run the Application

```bash
python -m app.main
```

---

## 💬 Example Queries

### ✅ Normal Queries

* What is the refund policy?
* How long does delivery take?
* Can I return a product?

### ⚠️ Sensitive Queries (HITL Trigger)

* I want to file a complaint
* I want legal action
* This service is terrible

---

## 👨‍💼 Human-in-the-Loop (HITL)

For sensitive or low-confidence queries, the system prompts:

```plaintext
Enter human support response:
```

This simulates escalation to a human support agent.

---

## 🧠 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Vector Databases (ChromaDB)
* Embeddings (HuggingFace Transformers)
* LangGraph Workflow Orchestration
* Conditional Routing
* Human-in-the-Loop (HITL)

---

## ⚠️ Challenges & Trade-offs

* Balancing **confidence threshold vs accuracy**
* Handling **sensitive queries safely**
* Trade-off between **chunk size and retrieval quality**
* Retrieval vs generation performance

---

## 🔮 Future Enhancements

* 🤖 Add LLM-based answer generation (GPT / Llama)
* 📊 Confidence scoring using similarity metrics
* 🌐 Streamlit / Web UI
* 📚 Multi-document support
* 🧠 Conversational memory
* ☁️ Deployment (AWS / Render)

---

## 📌 Conclusion

This project demonstrates a **production-style RAG system** with:

* Structured workflow orchestration
* Intelligent routing
* Safe handling via HITL

---

## 👤 Author

**Swathi P V**

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
