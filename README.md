# 📄 Local RAG Assistant

A privacy-first document analysis tool that runs entirely on your local machine. This application allows users to chat with their PDF documents without sending data to external cloud providers.

![Demo](https://via.placeholder.com/800x400?text=Insert+Your+Demo+GIF+Here)

## 🚀 Key Features
- **Zero Data Egress:** All processing (Embedding, Retrieval, Generation) happens locally.
- **Llama 3 Powered:** Uses the state-of-the-art open-source model via Ollama.
- **Fast Retrieval:** Implements ChromaDB with Nomic embeddings for sub-second query latency.
- **Interactive UI:** Built with Streamlit for a chat-like experience with session memory.

## 🛠️ Tech Stack
- **LLM:** Llama 3 (via Ollama)
- **Embeddings:** Nomic-Embed-Text / HuggingFace
- **Vector Database:** ChromaDB
- **Orchestration:** LangChain (LCEL)
- **Frontend:** Streamlit

## ⚙️ Installation & Setup

**Prerequisites:**
1. Install [Ollama](https://ollama.com).
2. Pull the required models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
