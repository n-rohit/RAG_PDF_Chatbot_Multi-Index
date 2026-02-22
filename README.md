# RAG Q&A Chatbot - Vector, Summary & Tree Indices

A production-ready **Retrieval Augmented Generation (RAG)** chatbot that loads PDF documents and answers questions using three different indexing strategies. Built with LlamaIndex, Ollama, ChromaDB, and Gradio.

---

## 🎯 What Is This Chatbot?

This RAG (Retrieval Augmented Generation) chatbot lets you **upload PDF documents** and **ask questions** about their content using AI. Instead of reading hundreds of pages manually, you can:

- Upload any PDF files (research papers, policies, manuals, reports)
- Ask natural language questions
- Get accurate answers backed by the actual document content
- Choose between three different search strategies for optimal results

**How it works:**
1. **Upload PDFs** through the web interface
2. **System processes** them using AI embeddings and indexing
3. **Ask questions** in plain English
4. **Get answers** with relevant context from your documents

**Example Use Cases:**
- "What are the company's core values?" (from company handbook)
- "Summarize the key findings of this research paper"
- "What is the refund policy?" (from terms & conditions)
- "List all safety procedures mentioned in this manual"

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
pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chroma

# Database & Embeddings
pip install chromadb sentence-transformers

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

Edit configuration variables at the top of the Python files:

```python
# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistral:latest"  # Ollama model to use
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
DOCS_FOLDER = "docs"  # Folder containing PDF files (rag_chatbot_fixed_docs.py only)
CHROMA_DB_PATH = "./chroma_db"  # Vector database location
TEMPERATURE = 0.7  # LLM creativity (0=factual, 1=creative)
TOP_K = 3  # Number of chunks to retrieve for Vector index
```

---

## 🚀 Usage

### Two Implementation Options

This project includes **two versions** of the chatbot, each suited for different workflows:

---

### **Option 1: `rag_chatbot.py` - Upload PDFs at Runtime**

**Use this when:** You want to upload different PDFs each time without restarting the app.

**Features:**
- ✅ Upload PDFs through the web interface
- ✅ Process multiple PDFs on-demand
- ✅ No need to restart the app for new documents
- ✅ Perfect for exploring different document sets
- ✅ Fresh ChromaDB collection created per upload session

**How to run:**

```bash
# Step 1: Make sure Ollama is running
ollama serve  # In a separate terminal

# Step 2: Start the chatbot
python3 rag_chatbot.py

# Step 3: Open browser
# Navigate to: http://127.0.0.1:7860

# Step 4: Use the interface
# - Click "Upload PDF files" button
# - Select one or more PDF files
# - Click "Process PDFs" button
# - Wait for indexing to complete
# - Enter your question
# - Select index type (Vector/Summary/Tree)
# - Click "Ask Question"
```

**Workflow:**
```
Launch App → Upload PDFs → Click "Process PDFs" → Wait → Ask Questions
```

**Best for:**
- Trying different document sets
- Quick experiments
- Demo/presentation scenarios
- One-time document analysis

---

### **Option 2: `rag_chatbot_fixed_docs.py` - Pre-load PDFs from Folder**

**Use this when:** You have a fixed set of PDFs that you want always available.

**Features:**
- ✅ Pre-indexes PDFs from `docs/` folder at startup
- ✅ Faster queries (no upload wait time)
- ✅ Persistent ChromaDB storage
- ✅ Perfect for production deployments
- ✅ Documents always ready to query

**How to run:**

```bash
# Step 1: Create docs folder and add PDFs
mkdir docs
cp your-document.pdf docs/
# Add all your PDF files to this folder

# Step 2: Make sure Ollama is running
ollama serve  # In a separate terminal

# Step 3: Start the chatbot
python3 rag_chatbot_fixed_docs.py

# Step 4: Wait for indexing
# The app will automatically:
# - Load all PDFs from docs/
# - Create Vector, Summary, and Tree indices
# - Display "✅ ALL 3 INDICES CREATED SUCCESSFULLY!"

# Step 5: Open browser
# Navigate to: http://127.0.0.1:7860

# Step 6: Ask questions immediately
# - Enter your question
# - Select index type (Vector/Summary/Tree)
# - Click "Ask Question"
```

**Workflow:**
```
Add PDFs to docs/ → Launch App → Automatic Indexing → Ready to Query
```

**Best for:**
- Production deployments
- Fixed document sets (company policies, manuals)
- Faster query response time
- Persistent knowledge base

---

### **Which Should I Use?**

| Scenario | Use This File | Why |
|----------|--------------|-----|
| Testing different documents | `rag_chatbot.py` | Upload flexibility |
| Quick demos/presentations | `rag_chatbot.py` | No setup needed |
| Fixed company documents | `rag_chatbot_fixed_docs.py` | Pre-indexed, faster |
| Production deployment | `rag_chatbot_fixed_docs.py` | Persistent storage |
| Exploring RAG concepts | `rag_chatbot.py` | Interactive learning |
| Building knowledge base | `rag_chatbot_fixed_docs.py` | Startup automation |

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

### Issue: "No PDF files found" (rag_chatbot_fixed_docs.py only)

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

### Issue: Timeout errors during processing

**Solution:**
The code already includes increased timeouts:
```python
# In both files, Ollama request_timeout is set to 3600.0 seconds (1 hour)
llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    request_timeout=3600.0,  # No timeouts for long documents
    temperature=TEMPERATURE,
)
```

If you still experience timeouts:
- Check Ollama server logs: `ollama logs`
- Reduce document size or number of documents
- Use Vector index for faster processing

---

## 📝 Project Structure

```
rag-chatbot/
├── rag_chatbot.py                    # Upload PDFs at runtime
├── rag_chatbot_fixed_docs.py         # Pre-load PDFs from folder
├── requirements.txt                  # Python dependencies
├── docs/                             # PDF documents folder (for fixed_docs.py)
│   ├── document1.pdf
│   └── document2.pdf
├── chroma_db/                        # Vector database (auto-created)
├── embeddings_cache/                 # Embedding models cache (auto-created)
└── README.md                         # This file
```

---

## 📚 Dependencies Explained

| Package | Version | Purpose |
|---------|---------|------------|
| `llama-index-core` | latest | RAG framework & indices |
| `chromadb` | latest | Vector database |
| `ollama` | latest | Local LLM inference |
| `sentence-transformers` | latest | Text embeddings |
| `pypdf` | latest | PDF parsing |
| `gradio` | latest | Web UI framework |
| `torch` | latest | Deep learning backend |
| `transformers` | latest | Pre-trained models |

---

## ✨ Features

- **3 Concurrent Indexing Strategies** - Compare approaches side-by-side
- **Two Usage Modes** - Upload-on-demand OR pre-indexed documents
- **Local LLM** - Mistral-7B via Ollama (no API keys needed)
- **Persistent Storage** - ChromaDB for vector embeddings
- **PDF Support** - Load and process PDF documents automatically
- **Clean UI** - Gradio web interface for easy interaction
- **Real-time Responses** - Instant query processing
- **Production Ready** - Comprehensive comments throughout
- **No Timeouts** - 1-hour request timeout for long documents

---

**Happy RAG-ing! 🚀**
