# 🤖 GraphBot — Chat with Your Documents using GraphRAG + Gemini

## 🔗 Live Demo - > 🌐 [Live]((https://graphrag-system.streamlit.app))

A smart chatbot powered by **Neo4j knowledge graphs**, **semantic vector search (FAISS)**, and **Google Gemini**.  
Built using `Streamlit` for interactive querying.

<img width="1774" height="943" alt="image" src="https://github.com/user-attachments/assets/15c9101e-5a50-4cc0-98e1-e86c01d6a886" />

## 🚀 Features

- 🔗 Graph-based relationship modeling using **Neo4j**
- 🔍 Semantic chunking and vector search using **FAISS**
- 🧠 Natural language answers with **Gemini LLM**
- 📄 Input from raw text or documents (PDF support coming soon)
- 💬 Clean chatbot UI with **Streamlit**

  ## 📦 Tech Stack

| Layer      | Tool |
|------------|------|
| LLM        | [Gemini 2.5 (via Google Generative AI)](https://ai.google.dev/)
| Vector DB  | [FAISS](https://github.com/facebookresearch/faiss)
| Graph DB   | [Neo4j](https://neo4j.com/)
| UI         | [Streamlit](https://streamlit.io/)
| Embedding  | Gemini `embedding-001`
| Parser     | LangChain-style chunker (optional)

---

## 📁 Project Structure

graphbot/
  ├── main.py # Backend logic: graph, embeddings, FAISS, Gemini
  ├── streamlit_app.py # Frontend chatbot using Streamlit
  ├── requirements.txt
  ├── .env (private)
  └── README.md

## Run the chatbot
- streamlit run streamlit_app.py

## How It Works
Paste input text (e.g. team structure, business logic, etc.)
Gemini extracts relationships → stored in Neo4j
Text is split into chunks → embedded using Gemini
FAISS indexes the chunks for fast retrieval
When you ask a question:
It retrieves both graph and vector context
Gemini answers using both

