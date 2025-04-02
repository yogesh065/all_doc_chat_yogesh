import streamlit as st
import hashlib
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
import pytesseract
import fitz  # PyMuPDF
import pandas as pd
from pptx import Presentation
from docx import Document as DocxDocument
from PIL import Image
import cv2
import numpy as np

# --------------------------
# Configuration & Constants
# --------------------------
MAX_THREADS = 4
SUPPORTED_EXTS = [
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
]
MAX_CHAT_HISTORY = 5  # Limit chat history to the last 5 messages

# Initialize components with caching
@st.cache_resource
def configure_system():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    groq_llm = Groq(model="llama-3.3-70b-specdec", api_key=st.secrets["k"]["api_key"])
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adjust path for your system
    return embed_model, groq_llm

embed_model, groq_llm = configure_system()
Settings.embed_model = embed_model
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# --------------------------
# Session State Management
# --------------------------
def initialize_session():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

def reset_chat_history():
    """Clear chat history while preserving other session state variables."""
    st.session_state.messages = []

# --------------------------
# File Processing Functions
# --------------------------
def preprocess_image(image_data):
    """Preprocess image for OCR."""
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.medianBlur(img, 3)

def extract_text_from_image(image_data):
    """Extract text from an image using Tesseract OCR."""
    try:
        processed_img = preprocess_image(image_data)
        return pytesseract.image_to_string(processed_img).strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def process_pdf(file_path):
    """Extract text and images from a PDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
            for img in page.get_images(full=True):
                base_image = doc.extract_image(img[0])
                text += "\n[IMAGE]: " + extract_text_from_image(base_image["image"])
    return text

def process_word(file_path):
    """Extract text and images from a Word document."""
    text = ""
    try:
        doc = DocxDocument(file_path)
        text += "\n".join([para.text for para in doc.paragraphs])
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_data = rel.target_part.blob
                text += "\n[IMAGE]: " + extract_text_from_image(img_data)
    except Exception as e:
        st.error(f"Word processing error: {str(e)}")
    return text

def process_excel(file_path):
    """Extract data from Excel files."""
    text = ""
    try:
        sheets = pd.ExcelFile(file_path).sheet_names
        for sheet_name in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            text += f"\nSheet: {sheet_name}\n{df.to_string(index=False)}\n"
    except Exception as e:
        st.error(f"Excel processing error: {str(e)}")
    return text

def process_powerpoint(file_path):
    """Extract text and images from a PowerPoint presentation."""
    text = ""
    try:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text += shape.text_frame.text + "\n"
                elif shape.shape_type == 13:  # Picture type
                    img_data = shape.image.blob
                    text += "\n[IMAGE]: " + extract_text_from_image(img_data)
    except Exception as e:
        st.error(f"PowerPoint processing error: {str(e)}")
    return text

def process_file(uploaded_file, temp_dir):
    """Route file to appropriate processing function."""
    file_path = Path(temp_dir) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    
    ext = file_path.suffix.lower()
    
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
    
    st.error(f"Unsupported file type: {ext}")
    return ""

@st.cache_resource(show_spinner=False)
def create_vector_index(content):
    """Create a vector index for querying document content."""
    return VectorStoreIndex.from_documents(
        [Document(text=content)],
        embed_model=embed_model,
        show_progress=True,
        insert_batch_size=512,
    ).as_query_engine(llm=groq_llm)

# --------------------------
# Chat Interface Functions
# --------------------------
def render_chat_messages():
    """Render chat messages in a scrollable container."""
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages[-MAX_CHAT_HISTORY:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

def handle_user_input():
    """Handle user input and generate responses."""
    
    if prompt := st.chat_input("Ask about your documents"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate assistant response using query engine
        with st.spinner("Analyzing..."):
            try:
                response = st.session_state.query_engine.query(prompt)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response,
                })
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

# --------------------------
# Main Application Layout
# --------------------------
def main():
