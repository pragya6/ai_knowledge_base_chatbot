# 🤖 AI Knowledge Base Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you chat with any webpage or PDF document. Paste a URL or upload a PDF — the app chunks, embeds, and indexes the content, then answers your questions grounded in the actual document with source citations.

**Live Demo:** https://ps06-aiknowledgebasechatbot.streamlit.app

---

## How It Works

```
User uploads a document (URL or PDF)
        ↓
Document is loaded and split into chunks
        ↓
Chunks are embedded and stored in ChromaDB (vector database)
        ↓
User asks a question
        ↓
Semantic search finds the most relevant chunks
        ↓
Relevant chunks + question are sent to Gemini LLM
        ↓
LLM generates an answer grounded in the document, with citations
```

## Architecture

| Component       | Technology                     | Purpose                                    |
|-----------------|--------------------------------|--------------------------------------------|
| LLM             | Google Gemini 2.5 Flash        | Answer generation                          |
| Vector Database | ChromaDB                       | Embedding storage and semantic search      |
| Framework       | LangChain                      | Prompt templates, chains, document loaders |
| Text Splitting  | RecursiveCharacterTextSplitter | Intelligent document chunking              |
| UI              | Streamlit                      | Interactive chat interface                 |


## Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API key (https://aistudio.google.com)

### Setup

```bash
# Clone the repository
git clone https://github.com/pragya6/knowledge-base-chatbot.git
cd knowledge-base-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate        # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# Run the app
streamlit run app.py
```

### Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Choose input type in the sidebar — **URL** or **PDF Upload**
3. Paste a URL or upload a PDF file
4. Click **Process** — the document is chunked, embedded, and indexed
5. Ask questions in the chat — answers are grounded in your document


## Limitations

- **In-memory vector store** — ChromaDB runs in-memory, so the knowledge base resets when the app restarts. For persistence, configure ChromaDB with a storage path.
- **Token limits** — very large documents may exceed the LLM's context window. The chunking and retrieval pipeline mitigates this by selecting only relevant chunks.
- **Free tier rate limits** — Gemini API free tier has per-minute and daily request limits.
- **Single document** — currently processes one document at a time. Processing a new document replaces the previous knowledge base.

## Future Improvements

- [ ] Persistent vector storage (ChromaDB with disk storage)
- [ ] Multi-document knowledge base
- [ ] Conversation memory with LangChain
- [ ] FastAPI backend for integration with any frontend
- [ ] Streaming responses for better UX

## Built With

- [LangChain](https://python.langchain.com/) — LLM application framework
- [ChromaDB](https://www.trychroma.com/) — Open-source vector database
- [Google Gemini](https://aistudio.google.com/) — Large language model
- [Streamlit](https://streamlit.io/) — Python web app framework

## License

MIT
