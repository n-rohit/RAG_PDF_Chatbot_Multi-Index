import os
import gradio as gr
import warnings
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
import chromadb

# LlamaIndex imports - monolithic style (0.9.48)
from llama_index import (
    Document,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
)

# LlamaIndex vector store and embedding imports
from llama_index.vector_stores import ChromaVectorStore  # ChromaDB vector store
from llama_index.llms.ollama import Ollama  # Ollama LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace embeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistral:latest"  # Ollama model
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
DOCS_FOLDER = "docs"  # PDF folder
CHROMA_DB_PATH = "./chroma_db"  # ChromaDB path
TEMPERATURE = 0.7  # LLM creativity
TOP_K = 3  # Vector retrieval count

# ============================================================================
# PART 1: INITIALIZE LLAMAINDEX
# ============================================================================

def initialize_settings():
    """Initialize LLM and embeddings"""
    
    print("🚀 Initializing LlamaIndex Settings...")
    
    # Create Ollama LLM instance
    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=TEMPERATURE,
    )
    
    # Create embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_folder="./embeddings_cache"
    )
    
    # Create service context (0.9.48 version)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    
    print("✅ Settings initialized!")
    print(f"   - LLM: {MODEL_NAME}")
    print(f"   - Embeddings: {EMBEDDING_MODEL}")
    
    return llm, embed_model, service_context

# ============================================================================
# PART 2: SETUP CHROMADB
# ============================================================================

def setup_chromadb():
    """Initialize ChromaDB vector store"""
    
    print("🗄️  Setting up ChromaDB...")
    
    # Create persistence directory
    Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name="rag_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    print(f"✅ ChromaDB initialized at: {CHROMA_DB_PATH}")
    
    return vector_store

# ============================================================================
# PART 3: LOAD PDF DOCUMENTS
# ============================================================================

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Load PDF documents"""
    
    print(f"📄 Loading PDF documents from {folder_path}...")
    
    # Create folder if needed
    if not Path(folder_path).exists():
        Path(folder_path).mkdir(exist_ok=True)
        print(f"⚠️  Folder created: {folder_path}")
        return []
    
    documents = []
    
    # Find PDF files
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found")
        return []
    
    # Load each PDF
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            text = ""
            
            # Extract text from all pages
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # Create Document
            doc = Document(
                text=text,
                metadata={"file_name": pdf_file.name}
            )
            documents.append(doc)
            print(f"   ✅ Loaded: {pdf_file.name}")
            
        except Exception as e:
            print(f"   ⚠️  Error reading {pdf_file.name}: {e}")
    
    print(f"✅ Loaded {len(documents)} PDF(s)")
    return documents

# ============================================================================
# PART 4: CREATE VECTOR INDEX
# ============================================================================

def create_vector_store_index(documents: List[Document], vector_store, service_context) -> VectorStoreIndex:
    """Create VectorStoreIndex"""
    
    print("\n📊 Creating VectorStoreIndex...")
    
    # Create storage context with vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True
    )
    
    print("✅ VectorStoreIndex created!")
    return index

# ============================================================================
# PART 5: CREATE SUMMARY INDEX
# ============================================================================

def create_summary_index(documents: List[Document], service_context) -> SummaryIndex:
    """Create SummaryIndex"""
    
    print("\n📋 Creating SummaryIndex...")
    
    index = SummaryIndex.from_documents(
        documents=documents,
        service_context=service_context,
        show_progress=True
    )
    
    print("✅ SummaryIndex created!")
    return index

# ============================================================================
# PART 6: CREATE TREE INDEX
# ============================================================================

def create_tree_index(documents: List[Document], service_context) -> TreeIndex:
    """Create TreeIndex"""
    
    print("\n🌳 Creating TreeIndex...")
    
    index = TreeIndex.from_documents(
        documents=documents,
        service_context=service_context,
        show_progress=True
    )
    
    print("✅ TreeIndex created!")
    return index

# ============================================================================
# PART 7: RAG CHATBOT
# ============================================================================

class RAGChatbot:
    """RAG Chatbot with all 3 indices"""
    
    def __init__(self, llm, embed_model, vector_store, service_context):
        self.llm = llm
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.service_context = service_context
        self.documents = []
        self.indices = {}
        
    def load_and_index_documents(self):
        """Load PDFs and create all indices"""
        
        # Load documents
        self.documents = load_documents_from_folder(DOCS_FOLDER)
        
        if not self.documents:
            print("❌ No PDFs found")
            return False
        
        try:
            # Create all three indices
            self.indices["Vector"] = create_vector_store_index(
                self.documents,
                self.vector_store,
                self.service_context
            )
            
            self.indices["Summary"] = create_summary_index(
                self.documents,
                self.service_context
            )
            
            self.indices["Tree"] = create_tree_index(
                self.documents,
                self.service_context
            )
            
            print("\n" + "="*60)
            print("✅ ALL 3 INDICES CREATED SUCCESSFULLY!")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def query(self, question: str, index_type: str = "Vector") -> str:
        """Query using specified index"""
        
        if not self.documents:
            return "❌ No documents loaded"
        
        if index_type not in self.indices:
            return f"❌ Invalid index: {index_type}"
        
        try:
            selected_index = self.indices[index_type]
            
            # Create query engine
            if index_type == "Vector":
                query_engine = selected_index.as_query_engine(
                    similarity_top_k=TOP_K,
                    service_context=self.service_context
                )
            elif index_type == "Summary":
                query_engine = selected_index.as_query_engine(
                    service_context=self.service_context
                )
            else:  # Tree
                query_engine = selected_index.as_query_engine(
                    service_context=self.service_context
                )
            
            # Execute query
            print(f"\n🔍 Querying {index_type} index...")
            response = query_engine.query(question)
            
            return str(response)
            
        except Exception as e:
            return f"❌ Query error: {str(e)}"

# ============================================================================
# PART 8: GRADIO UI
# ============================================================================

def create_gradio_ui(chatbot: RAGChatbot):
    """Create Gradio interface"""
    
    def answer_question(question: str, index_type: str) -> Tuple[str, str]:
        """Handle question"""
        
        if not question.strip():
            return "Please enter a question", "Empty question"
        
        # Query chatbot
        answer = chatbot.query(question, index_type)
        
        # Info display
        info = f"**Index:** {index_type}\n**Question:** {question[:80]}..."
        
        return answer, info
    
    # Create UI
    with gr.Blocks(theme=gr.themes.Soft(), title="RAG Q&A Chatbot") as demo:
        
        # Header section
        gr.Markdown(" ")
        gr.Markdown(" ")
        gr.Markdown(" ")
        gr.Markdown(" ")
        gr.Markdown(" ")
        gr.Markdown("# 💬 RAG Q&A Chatbot")
        gr.Markdown("## Ask Questions from Loaded Documents")
        gr.Markdown(" ")
        gr.Markdown(" ")
        gr.Markdown(" ")
        
        # Main content
        with gr.Row():
            with gr.Column(scale=1):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about your PDFs...",
                    lines=3
                )
                
                # Index selector
                index_selector = gr.Radio(
                    choices=["Vector", "Summary", "Tree"],
                    value="Vector",
                    label="Index Type",
                    info="Vector: Fast | Summary: Deep | Tree: Hierarchical"
                )
                
                # Submit button
                submit_btn = gr.Button("Ask Question", variant="primary")
            
            with gr.Column(scale=2):
                # Answer output
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=12,
                    interactive=False
                )
                
                # Info output
                info_output = gr.Markdown(
                    value="Ready for questions!",
                    label="Info"
                )
        
        # Connect buttons
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, index_selector],
            outputs=[answer_output, info_output]
        )
        
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, index_selector],
            outputs=[answer_output, info_output]
        )
    
    return demo

# ============================================================================
# PART 9: MAIN
# ============================================================================

def main():
    """Main startup"""
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     PDFs RAG Q&A CHATBOT - Vector, Summary, Tree     ║
    ║     (3 Indices - 3 Data Organization Techniques)     ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize
        llm, embed_model, service_context = initialize_settings()
        
        # Setup ChromaDB
        vector_store = setup_chromadb()
        
        # Create chatbot
        chatbot = RAGChatbot(llm, embed_model, vector_store, service_context)
        
        # Load and index
        if not chatbot.load_and_index_documents():
            print("⚠️  No PDFs found. Add them to 'docs/' folder and restart.")
        
        # Launch UI
        print("\n🌐 Launching Gradio...")
        print("📱 Open: http://127.0.0.1:7860")
        
        demo = create_gradio_ui(chatbot)
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            share=False,
            debug=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
