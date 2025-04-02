import streamlit as st
import fitz
import pandas as pd
import pytesseract
import textract
from pathlib import Path
from PIL import Image
import io
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import tempfile
import shutil
import warnings
import hashlib
from openpyxl import load_workbook
from pptx import Presentation

# Suppress all warnings before other imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="huggingface_hub")

# --------------------------
# Configuration & Constants
# --------------------------
MAX_THREADS = 4
OCR_DPI = 300
SUPPORTED_EXTS = [
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".rtf", ".txt", ".csv", ".json", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
]

# Initialize components with caching
@st.cache_resource
def configure_system():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    groq_llm = Groq(model="llama-3.3-70b-specdec", api_key=st.secrets["k"]["api_key"])
    
    # Configure Tesseract
    tesseract_path = shutil.which('tesseract')
    if not tesseract_path:
        raise EnvironmentError('Tesseract not found in system PATH')
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    return embed_model, groq_llm

embed_model, groq_llm = configure_system()
Settings.embed_model = embed_model
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# --------------------------
# Optimized File Processing
# --------------------------
def file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getbuffer()).hexdigest()

def process_excel(file_path):
    """Optimized Excel processing with chunked reading"""
    text = ""
    try:
        # Read metadata first
        sheets = pd.ExcelFile(file_path).sheet_names
        
        # Process each sheet with chunking
        for sheet_name in sheets:
            chunks = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                engine='openpyxl',
                chunksize=1000
            )
            sheet_text = f"\nSheet: {sheet_name}\n"
            for chunk in chunks:
                sheet_text += chunk.to_string(index=False) + "\n"
            text += sheet_text

        # Image extraction for .xlsx
        if file_path.suffix.lower() == ".xlsx":
            wb = load_workbook(file_path, read_only=True)
            for sheet in wb.worksheets:
                for image in sheet._images:
                    img_data = image._data()
                    text += "\n[IMAGE]: " + extract_text_from_image(img_data)
                    
        return text
    except Exception as e:
        st.error(f"Excel processing error: {str(e)}")
        return ""

@st.cache_data(max_entries=5, ttl=3600, show_spinner=False)
def process_file(uploaded_file, temp_dir):
    """Unified file processor with optimized caching"""
    file_path = Path(temp_dir) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    
    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        return process_pdf(file_path)
    elif ext in (".xlsx", ".xls"):
        return process_excel(file_path)
    elif ext in (".pptx", ".ppt", ".doc", ".docx"):
        return process_office(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        return extract_text_from_image(file_path.read_bytes())
    else:
        try:
            return textract.process(str(file_path)).decode("utf-8")
        except:
            return uploaded_file.getvalue().decode("utf-8", errors="replace")

# --------------------------
# Session State Management
# --------------------------
def reset_session():
    st.session_state.messages = []
    st.session_state.last_file_hash = None
    st.session_state.query_engine = None

@st.cache_resource(show_spinner=False)
def create_vector_index(content):
    """Cached index creation based on content hash"""
    return VectorStoreIndex.from_documents(
        [Document(text=content)],
        embed_model=embed_model
    ).as_query_engine(llm=groq_llm)

# --------------------------
# Main Application
# --------------------------
def main():
    st.title("📄 Universal Document Processor")
    
    # Initialize session state
    if "messages" not in st.session_state:
        reset_session()

    # File upload with session reset
    uploaded_files = st.file_uploader(
        "Upload Documents/Images",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
        on_change=reset_session
    )

    if uploaded_files:
        current_hash = hashlib.md5(b''.join(
            f.getbuffer() for f in uploaded_files
        )).hexdigest()
        
        if current_hash != st.session_state.get("last_file_hash"):
            with st.spinner("Processing Files..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                            processed_content = list(executor.map(
                                lambda f: process_file(f, temp_dir),
                                uploaded_files
                            ))
                        
                        combined_content = "\n\n".join(processed_content)
                        st.session_state.query_engine = create_vector_index(combined_content)
                        st.session_state.last_file_hash = current_hash
                        
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    reset_session()

    # Chat interface
    if st.session_state.query_engine:
        st.header("💬 Document Insights")
        
        for msg in st.session_state.messages[-5:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.query_engine.query(prompt)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.response
                    })
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

if __name__ == "__main__":
    main()
