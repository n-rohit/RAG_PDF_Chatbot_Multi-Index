# RAG Q&A Chatbot - Vector, Summary & Tree Indices

A production-ready **Retrieval Augmented Generation (RAG)** chatbot that loads PDF documents and answers questions using three different indexing strategies. Built with LlamaIndex, Ollama, ChromaDB, and Gradio.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Three Index Types Explained](#three-index-types-explained)
3. [When to Use Each Index](#when-to-use-each-index)
4. [Features](#features)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Architecture](#architecture)
9. [Gradio UI vs Web Frameworks](#gradio-ui-vs-web-frameworks)
10. [Screenshots](#screenshots)
11. [Example Queries](#example-queries)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This RAG chatbot processes PDF documents and generates intelligent answers by combining:
- **Vector Search** (Embedding-based similarity)
- **Summary Index** (Sequential document analysis)
- **Tree Index** (Hierarchical document structure)

Each index approaches document retrieval differently, allowing you to choose the best strategy for your use case.

---

## 📚 Three Index Types Explained

### 1. **Vector Index (Fast Semantic Search)**

**How it works:**
- Converts documents into numerical embeddings (vectors) using HuggingFace embeddings
- Stores vectors in ChromaDB for fast similarity search
- At query time, converts your question to a vector and finds most similar document chunks

**Example workflow:**
```
User Question: "What are the company's core values?"
    ↓
Convert to embedding (vector)
    ↓
Find top-3 most similar document chunks using cosine similarity
    ↓
Feed top chunks to LLM for final answer
    ↓
Answer: "The company's core values are integrity, respect, 
         fair operating practices, and citizenship."
```

**Characteristics:**
- ✅ **Speed:** Fast (1-3 seconds)
- ✅ **Best for:** General Q&A, quick lookups, long documents
- ✅ **Retrieval:** Top-K similar chunks (default: 3)
- ❌ **Cons:** May miss context from non-similar sections

---

### 2. **Summary Index (Sequential Deep Analysis)**

**How it works:**
- Loads entire documents sequentially as a list of nodes
- No embeddings required - works with raw text
- At query time, processes all chunks and synthesizes comprehensive answer

**Example workflow:**
```
User Question: "What are the company's core values?"
    ↓
Load ALL document chunks in sequence
    ↓
Create summaries of all chunks
    ↓
Synthesize answer from entire document context
    ↓
Answer: "Based on complete document analysis, the company's 
         core values span ethics, sustainability, employee 
         well-being, community responsibility, and 
         transparency in all operations..."
```

**Characteristics:**
- ✅ **Depth:** Analyzes entire document holistically
- ✅ **Best for:** Complete document analysis, comprehensive summaries, reports
- ✅ **Accuracy:** High (uses all available information)
- ❌ **Speed:** Slower for large documents (10-30 seconds)

---

### 3. **Tree Index (Hierarchical Navigation)**

**How it works:**
- Builds hierarchical tree structure of document summaries
- Parent nodes summarize groups of child nodes
- At query time, traverses tree top-down to find most relevant leaf nodes

**Example workflow:**
```
Document Structure:
├── Level 1 (Root): "Complete company document"
├── Level 2: ["Ethics & Values", "Operations", "Sustainability"]
├── Level 3: ["Integrity", "Respect", "Fair Practices"], 
             ["Efficiency", "Innovation"], 
             ["Environmental", "Social"]
└── Level 4: [Specific details...]

User Question: "What are the company's core values?"
    ↓
Start at root: "Is this about company overview?"
    ↓
Navigate to Level 2: "Likely in Ethics & Values branch"
    ↓
Navigate to Level 3: Check "Integrity", "Respect", "Fair Practices"
    ↓
Retrieve leaf nodes with specific information
    ↓
Answer: "Core values include integrity, respect for individuals,
         fair operating practices, and active citizenship."
```

**Characteristics:**
- ✅ **Balance:** Fast + contextual (5-10 seconds)
- ✅ **Best for:** Structured documents (reports, policies, guides)
- ✅ **Navigation:** Hierarchical exploration of content
- ✅ **Recommended:** For most use cases

---

## 🎛️ When to Use Each Index

| Use Case | Recommended Index | Why |
|----------|------------------|-----|
| **Quick fact lookup** | Vector | Fastest for simple questions |
| **Specific data point** | Vector | Directly retrieves relevant chunks |
| **Full document summary** | Summary | Analyzes all content |
| **Comprehensive analysis** | Summary | Best for reports and detailed answers |
| **Policy documents** | Tree | Structured, hierarchical nature |
| **Research papers** | Tree | Sections and subsections work well |
| **General Q&A** | Tree | Best balance of speed and accuracy |
| **Large documents (50+ pages)** | Vector | Manages scale efficiently |
| **Small focused documents** | Summary | No performance penalty, high quality |

---

## ✨ Features

- **3 Concurrent Indexing Strategies** - Compare approaches side-by-side
- **Local LLM** - Mistral-7B via Ollama (no API keys needed)
- **Persistent Storage** - ChromaDB for vector embeddings
- **PDF Support** - Load and process PDF documents automatically
- **Clean UI** - Gradio web interface for easy interaction
- **Real-time Responses** - Instant query processing
- **Production Ready** - Comments throughout for understanding

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- Ollama installed ([Download here](https://ollama.ai))
- Mistral-7B model: `ollama pull mistral:latest`

### Step 1: Clone/Download Project

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

**Option A: From requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual installation**
```bash
# Core RAG Framework
pip install llama-index==0.9.48 chromadb==0.4.24

# LLM & Embeddings
pip install ollama sentence-transformers

# PDF Processing
pip install pypdf

# Web UI
pip install gradio

# ML/NLP
pip install transformers torch
```

### Step 4: Verify Ollama is Running

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check if running
curl http://localhost:11434/api/tags

# Should return list of available models
```

---

## ⚙️ Configuration

Edit configuration variables in `rag_chatbot_vec_tree_summ.py`:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistral:latest"  # Ollama model to use
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
DOCS_FOLDER = "docs"  # Folder containing PDF files
CHROMA_DB_PATH = "./chroma_db"  # Vector database location
TEMPERATURE = 0.7  # LLM creativity (0=factual, 1=creative)
TOP_K = 3  # Number of chunks to retrieve for Vector index
```

---

## 🚀 Usage

### 1. Add PDF Documents

Create a `docs/` folder and add your PDF files:

```bash
mkdir docs
# Copy your PDF files to this folder
cp your-document.pdf docs/
```

### 2. Start the Application

```bash
# Make sure Ollama is running in another terminal
python3 rag_chatbot_vec_tree_summ.py
```

### 3. Open Gradio Interface

- Open browser to: `http://127.0.0.1:7860`
- You should see the RAG Q&A Chatbot UI

### 4. Ask Questions

1. Enter your question in the text box
2. Select index type (Vector, Summary, or Tree)
3. Click "Ask Question" or press Enter
4. View answer and query information

---

## 🏗️ Architecture

```
PDF Documents
    ↓
[Load & Parse]
    ↓
    ├─→ [Vector Index] ──→ ChromaDB
    ├─→ [Summary Index] ──→ Memory
    └─→ [Tree Index] ──→ Memory
    ↓
[Ollama LLM + HuggingFace Embeddings]
    ↓
[Gradio Web Interface]
    ↓
User Interaction
```

### Data Flow for Each Index Type:

**Vector Index Path:**
```
Question → HuggingFace Embedding → Vector Search in ChromaDB → Top-K chunks → Ollama LLM → Answer
```

**Summary Index Path:**
```
Question → Sequential Node Retrieval → All chunks loaded → Tree Summarization → Ollama LLM → Answer
```

**Tree Index Path:**
```
Question → Hierarchical Traversal → Relevant leaf nodes → Tree Summarization → Ollama LLM → Answer
```

---

## 🎨 Gradio UI vs Web Frameworks

### Why Gradio for Prototyping?

**Gradio Benefits:**
- ✅ **Fastest prototyping** - Build UI in minutes
- ✅ **No frontend knowledge needed** - Pure Python
- ✅ **Built-in components** - Buttons, inputs, outputs
- ✅ **Automatic documentation** - API auto-generated
- ✅ **Easy sharing** - Public links available

### Scaling to Production Web Apps

The chatbot logic can power **Flask, React, Angular, or Mobile Apps** by using the same backend with APIs:

**Architecture Pattern:**

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend Layer                          │
│  (React/Angular/Mobile/Web)                             │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────────────────────┐
│                  API Layer (Flask/FastAPI)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ @app.route('/api/query', methods=['POST'])        │  │
│  │ def query_chatbot():                              │  │
│  │     question = request.json['question']           │  │
│  │     index_type = request.json['index_type']       │  │
│  │     answer = chatbot.query(question, index_type)  │  │
│  │     return {'answer': answer}                     │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │ Python Method Calls
┌──────────────────▼──────────────────────────────────────┐
│              RAGChatbot Backend Logic                    │
│  (Same code from this project)                          │
│  - load_documents_from_folder()                         │
│  - create_vector_store_index()                          │
│  - create_summary_index()                               │
│  - create_tree_index()                                  │
│  - query()                                              │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│             Data Layer                                   │
│  - ChromaDB (vector storage)                            │
│  - Ollama LLM                                           │
│  - PDF Documents                                        │
└──────────────────────────────────────────────────────────┘
```

### Example: Converting to Flask API

**Original Gradio Code:**
```python
# rag_chatbot_vec_tree_summ.py
chatbot = RAGChatbot(llm, embed_model, vector_store, service_context)
answer = chatbot.query("What are values?", "Vector")
```

**Flask Wrapper (new file: `app.py`):**
```python
from flask import Flask, request, jsonify
from rag_chatbot_vec_tree_summ import RAGChatbot, initialize_settings, setup_chromadb

app = Flask(__name__)

# Initialize once at startup
llm, embed_model, service_context = initialize_settings()
vector_store = setup_chromadb()
chatbot = RAGChatbot(llm, embed_model, vector_store, service_context)
chatbot.load_and_index_documents()

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    """API endpoint for RAG queries"""
    data = request.json
    question = data.get('question')
    index_type = data.get('index_type', 'Vector')
    
    answer = chatbot.query(question, index_type)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'index_type': index_type
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**React Frontend (consume API):**
```javascript
async function askQuestion(question, indexType) {
    const response = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, index_type: indexType })
    });
    
    const data = await response.json();
    console.log('Answer:', data.answer);
    return data;
}

// Usage
askQuestion('What are company values?', 'Vector');
```

**This same chatbot backend powers:**
- ✅ Gradio UI (current)
- ✅ Flask web app
- ✅ FastAPI service
- ✅ React/Angular frontend
- ✅ Mobile app (via API)
- ✅ Slack bot
- ✅ Discord bot

---

## 📊 Screenshots

### 1. Gradio UI Output



---

### 2. Terminal Output



---

## 🔧 Troubleshooting

### Issue: "Cannot connect to Ollama"

```
❌ Error: Connection refused (Ollama not running)
```

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check connectivity
curl http://localhost:11434/api/tags
```

---

### Issue: "No PDF files found"

```
⚠️  No PDF files found in docs/ folder
```

**Solution:**
```bash
mkdir docs
# Add your PDF files to docs/ folder
ls docs/  # Verify files are there
```

---

### Issue: "Out of memory" during index creation

**Solution:**
- Use Vector Index only (fastest)
- Reduce document size
- Close other applications
- Use a machine with more RAM

---

## 📝 Project Structure

```
rag-chatbot/
├── rag_chatbot_vec_tree_summ.py    # Main application
├── requirements.txt                  # Python dependencies
├── docs/                             # PDF documents folder
│   ├── document1.pdf
│   └── document2.pdf
├── chroma_db/                        # Vector database (auto-created)
├── embeddings_cache/                 # Embedding models cache (auto-created)
└── README.md                         # This file
```

---

## 📚 Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| `llama-index` | 0.9.48 | RAG framework & indices |
| `chromadb` | 0.4.24 | Vector database |
| `ollama` | latest | Local LLM inference |
| `sentence-transformers` | 2.2.0+ | Text embeddings |
| `pypdf` | 4.0.0+ | PDF parsing |
| `gradio` | 4.0.0+ | Web UI framework |
| `torch` | 2.0.0+ | Deep learning backend |
| `transformers` | 4.30.0+ | Pre-trained models |

---
**Happy RAG-ing! 🚀**
