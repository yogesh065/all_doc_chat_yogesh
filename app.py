import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
import textract
from pathlib import Path
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
from docx import Document as DocxDocument
from PIL import Image
from streamlit.components.v1 import html

# Suppress all warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

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
# Core Processing Functions
# --------------------------
def preprocess_image(image_data):
    """Optimized image preprocessing for OCR"""
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.medianBlur(img, 3)

@st.cache_data(max_entries=10, persist="disk", show_spinner=False)
def extract_text_from_image(image_data):
    """Cached OCR text extraction"""
    try:
        processed_img = preprocess_image(image_data)
        return pytesseract.image_to_string(processed_img).strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def process_pdf(file_path):
    """PDF text and image extraction"""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
            for img in page.get_images(full=True):
                base_image = doc.extract_image(img[0])
                text += "\n[IMAGE]: " + extract_text_from_image(base_image["image"])
    return text

def process_word(file_path):
    """Word document processing with image extraction"""
    text = ""
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        # Extract images from Word document
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_data = rel.target_part.blob
                text += "\n[IMAGE]: " + extract_text_from_image(img_data)
    except Exception as e:
        st.error(f"Word processing error: {str(e)}")
    return text

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
    except Exception as e:
        st.error(f"Excel processing error: {str(e)}")
    return text

def process_powerpoint(file_path):
    """PowerPoint processing with image extraction"""
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text += shape.text_frame.text + "\n"
                elif shape.shape_type == 13:  # Picture
                    text += "\n[IMAGE]: " + extract_text_from_image(shape.image.blob)
    except Exception as e:
        st.error(f"PowerPoint processing error: {str(e)}")
    return text

# --------------------------
# Unified File Processor
# --------------------------
@st.cache_data(max_entries=5, ttl=3600, show_spinner=False)
def process_file(uploaded_file, temp_dir):
    """File processing router with optimized caching"""
    file_path = Path(temp_dir) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    ext = file_path.suffix.lower()

    try:
        if ext == ".pdf":
            return process_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return process_word(file_path)
        elif ext in (".xlsx", ".xls"):
            return process_excel(file_path)
        elif ext in (".pptx", ".ppt"):
            return process_powerpoint(file_path)
        elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            return extract_text_from_image(file_path.read_bytes())
        else:
            return textract.process(str(file_path)).decode("utf-8")
    except Exception as e:
        st.error(f"Error processing {file_path.name}: {str(e)}")
        return ""

# --------------------------
# Session Management
# --------------------------
def reset_session():
    st.session_state.messages = []
    st.session_state.last_file_hash = None
    st.session_state.query_engine = None

@st.cache_resource(show_spinner=False)
def create_vector_index(content):
    """Cached index creation with content-based invalidation"""
    return VectorStoreIndex.from_documents(
        [Document(text=content)],
        embed_model=embed_model
    ).as_query_engine(llm=groq_llm)

# --------------------------
# Auto-scroll Functionality
# --------------------------
def auto_scroll():
    scroll_js = """
    <script>
    function scrollToBottom() {
        window.parent.document.querySelectorAll(
            '[data-testid="stVerticalBlock"]'
        ).forEach(function(el) {
            if (el.scrollHeight > el.clientHeight) {
                el.scrollTop = el.scrollHeight;
            }
        });
    }
    setTimeout(scrollToBottom, 100);
    </script>
    """
    html(scroll_js, height=0)

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
                        reset_session()  # Clear previous chat history
                        
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    reset_session()

    # Chat interface
    if st.session_state.query_engine:
        st.header("💬 Document Insights")
        
        # Create chat container with max height
        chat_container = st.container(height=500)
        
        with chat_container:
            # Show last 5 messages
            for msg in st.session_state.messages[-5:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Auto-scroll to bottom after rendering messages
            auto_scroll()

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

            # Force UI update
            st.rerun()

if __name__ == "__main__":
    main()
