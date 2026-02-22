import os
import gradio as gr
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
from pypdf import PdfReader
import chromadb
import uuid

# LlamaIndex imports - monolithic style (0.9.48)
from llama_index.core import (
    Document,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

# LlamaIndex vector store and embedding imports
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
TEMPERATURE = 0.7
TOP_K = 3

# ============================================================================
# PART 1: INITIALIZE LLAMAINDEX
# ============================================================================

def initialize_settings():
    """Initialize LLM and embeddings (new Settings API)"""
    print("Initializing LlamaIndex Settings...")

    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=3600.0,
        temperature=TEMPERATURE,
    )

    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_folder="./embeddings_cache",
    )

    # New global configuration (replaces ServiceContext)
    Settings.llm = llm
    Settings.embed_model = embed_model

    print("Settings initialized!")
    print(f" - LLM: {MODEL_NAME}")
    print(f" - Embeddings: {EMBEDDING_MODEL}")

    return llm, embed_model

# ============================================================================
# PART 2: SETUP CHROMADB
# ============================================================================

def setup_chromadb_client():
    """Initialize ChromaDB persistent client (collections will be created per upload)."""
    print("Setting up ChromaDB client...")
    Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"ChromaDB ready at: {CHROMA_DB_PATH}")
    return chroma_client

def create_new_vector_store(chroma_client) -> ChromaVectorStore:
    """Create a fresh collection for the current upload session."""
    collection_name = f"rag_documents_{uuid.uuid4().hex[:8]}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return ChromaVectorStore(chroma_collection=collection)

# ============================================================================
# PART 3: LOAD PDF DOCUMENTS (FROM UPLOADS)
# ============================================================================

def load_documents_from_filepaths(filepaths: List[str], progress: Optional[gr.Progress] = None) -> List[Document]:
    """Load PDF documents from uploaded file paths."""
    documents: List[Document] = []

    if not filepaths:
        return documents

    total = len(filepaths)
    for i, fp in enumerate(filepaths):
        try:
            if progress is not None:
                progress((i / max(total, 1)) * 0.35, desc=f"Reading PDF {i+1}/{total}")

            reader = PdfReader(fp)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            doc = Document(
                text=text,
                metadata={"file_name": Path(fp).name, "source_path": fp},
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if progress is not None:
        progress(0.35, desc="Finished reading PDFs")

    return documents

# ============================================================================
# PART 4/5/6: CREATE INDICES
# ============================================================================

def create_vector_store_index(documents: List[Document], vector_store) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )
    return index

def create_summary_index(documents: List[Document]) -> SummaryIndex:
    return SummaryIndex.from_documents(
        documents=documents,
        show_progress=True,
    )

def create_tree_index(documents: List[Document]) -> TreeIndex:
    return TreeIndex.from_documents(
        documents=documents,
        show_progress=True,
    )

# ============================================================================
# PART 7: RAG CHATBOT
# ============================================================================

class RAGChatbot:
    def __init__(self, llm, embed_model, chroma_client):
        self.llm = llm
        self.embed_model = embed_model
        self.chroma_client = chroma_client
        self.documents: List[Document] = []
        self.indices = {}
        self.ready = False

    def ingest_uploaded_pdfs(self, pdf_filepaths: List[str], progress: Optional[gr.Progress] = None) -> Tuple[bool, str]:
        """Build all indices from uploaded PDFs."""
        self.ready = False
        self.indices = {}
        self.documents = []

        if not pdf_filepaths:
            return False, "No files provided."

        if progress is not None:
            progress(0.0, desc="Starting ingestion")

        # 1) Load documents
        self.documents = load_documents_from_filepaths(pdf_filepaths, progress=progress)
        if not self.documents:
            return False, "Could not read any PDFs (empty or failed extraction)."

        # 2) Create a fresh vector store/collection for this ingestion
        if progress is not None:
            progress(0.4, desc="Preparing vector store")
        vector_store = create_new_vector_store(self.chroma_client)

        # 3) Build indices
        try:
            if progress is not None:
                progress(0.5, desc="Building Vector index")
            self.indices["Vector"] = create_vector_store_index(self.documents, vector_store)

            if progress is not None:
                progress(0.7, desc="Building Summary index")
            self.indices["Summary"] = create_summary_index(self.documents)

            if progress is not None:
                progress(0.85, desc="Building Tree index")
            self.indices["Tree"] = create_tree_index(self.documents)

            self.ready = True
            if progress is not None:
                progress(1.0, desc="Ready")

            return True, f"Indexed {len(self.documents)} PDF(s). Ready to chat."

        except Exception as e:
            self.ready = False
            return False, f"Indexing error: {e}"

    def query(self, question: str, index_type: str = "Vector") -> str:
        if not self.ready:
            return "Upload PDFs and click 'Process PDFs' first."

        if not question.strip():
            return "Please enter a question."

        if index_type not in self.indices:
            return f"Invalid index: {index_type}"

        selected_index = self.indices[index_type]

        if index_type == "Vector":
            query_engine = selected_index.as_query_engine(similarity_top_k=TOP_K)
        else:
            query_engine = selected_index.as_query_engine()

        response = query_engine.query(question)
        return str(response)

# ============================================================================
# PART 8: GRADIO UI
# ============================================================================

def create_gradio_ui(chatbot: RAGChatbot):
    def process_pdfs(files: List[str], progress=gr.Progress()):
        """
        files is a list[str] of filepaths when using gr.File(type="filepath", file_count="multiple").
        """
        # Disable chat controls while processing; status shows progress via gr.Progress.
        status = "Processing..."
        ok, msg = chatbot.ingest_uploaded_pdfs(files or [], progress=progress)

        # Enable/disable chat based on readiness
        question_update = gr.update(value="", interactive=ok)
        ask_update = gr.update(interactive=ok)
        index_update = gr.update(interactive=ok)
        answer_update = gr.update(value="")
        info_update = gr.update(value=msg if ok else f"Not ready: {msg}")

        return msg if ok else f"Error: {msg}", question_update, ask_update, index_update, answer_update, info_update

    def answer_question(question: str, index_type: str):
        answer = chatbot.query(question, index_type)
        info = f"**Index:** {index_type}\n\n**Ready:** {chatbot.ready}"
        return answer, info

    with gr.Blocks(theme=gr.themes.Soft(), title="RAG Q&A Chatbot") as demo:
        gr.Markdown("# RAG Q&A Chatbot")
        gr.Markdown("Upload PDFs, process them, then ask questions.")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_uploader = gr.File(
                    label="Upload PDF files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )

                process_btn = gr.Button("Process PDFs", variant="primary")
                status_md = gr.Markdown("Status: waiting for PDFs...")

                index_selector = gr.Radio(
                    choices=["Vector", "Summary", "Tree"],
                    value="Vector",
                    label="Index Type",
                    info="Vector: retrieval | Summary: global summary | Tree: hierarchical",
                    interactive=False,
                )

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about your uploaded PDFs...",
                    lines=3,
                    interactive=False,
                )

                submit_btn = gr.Button("Ask Question", variant="secondary", interactive=False)

            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", lines=12, interactive=False)
                info_output = gr.Markdown(value="Upload PDFs and click Process PDFs.", label="Info")

        # Process PDFs with progress
        process_btn.click(
            fn=process_pdfs,
            inputs=[pdf_uploader],
            outputs=[status_md, question_input, submit_btn, index_selector, answer_output, info_output],
            show_progress="full",
        )

        # Ask question
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, index_selector],
            outputs=[answer_output, info_output],
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, index_selector],
            outputs=[answer_output, info_output],
        )

    # Needed so progress + spinner works reliably for longer runs
    demo.queue()
    return demo

# ============================================================================
# PART 9: MAIN
# ============================================================================

def main():
    print("Starting PDFs RAG Q&A Chatbot (Upload -> Process -> Chat)")
    llm, embed_model = initialize_settings()
    chroma_client = setup_chromadb_client()

    chatbot = RAGChatbot(llm, embed_model, chroma_client)

    demo = create_gradio_ui(chatbot)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=True,
        debug=True,
        prevent_thread_lock=False,
    )

if __name__ == "__main__":
    main()
