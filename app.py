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
from pptx import Presentation
# 1. Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
# --------------------------
# Configuration & Constants
# --------------------------
MAX_THREADS = 4
OCR_DPI = 300
SUPPORTED_EXTS = [
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".rtf", ".txt", ".csv", ".json", ".html", ".htm",
    # Images
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
# Enhanced OCR Processing
# --------------------------
def preprocess_image(image_data):
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return cv2.medianBlur(img, 3)

@st.cache_data(max_entries=10, persist="disk")
def extract_text_from_image(image_data):
    try:
        processed_img = preprocess_image(image_data)
        return pytesseract.image_to_string(processed_img).strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

# --------------------------
# Document Processing
# --------------------------
def process_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
            for img in page.get_images():
                base_image = doc.extract_image(img[0])
                text += "\n[IMAGE]: " + extract_text_from_image(base_image["image"])
    return text

def process_office(file_path):
    try:
        ext = file_path.suffix.lower()
        text = ""
        
        # PowerPoint processing
        if ext in (".pptx", ".ppt"):
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        text += shape.text_frame.text + "\n"
                    elif shape.shape_type == 13:  # Picture
                        text += "\n[IMAGE]: " + extract_text_from_image(shape.image.blob)
        
        # Word processing
        elif ext in (".docx", ".doc"):
            from docx import Document
            doc = Document(file_path)
            # Extract text
            text = "\n".join([para.text for para in doc.paragraphs])
            # Extract images
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    img_data = rel.target_part.blob
                    text += "\n[IMAGE]: " + extract_text_from_image(img_data)
        
        # Excel processing
        elif ext in (".xlsx", ".xls"):
            # Text extraction
            df = pd.read_excel(file_path, sheet_name=None)
            text = "\n".join([f"Sheet: {name}\n{df.to_string()}" 
                            for name, df in df.items()])
            # Image extraction
            if ext == ".xlsx":
                from openpyxl import load_workbook
                wb = load_workbook(file_path)
                for sheet in wb.worksheets:
                    for image in sheet._images:
                        img_data = image._data()
                        text += "\n[IMAGE]: " + extract_text_from_image(img_data)
        
        # Fallback for other formats
        else:
            text = textract.process(
                str(file_path),
                extension=ext.replace(".", "")
            ).decode("utf-8")
            
        return text
        
    except Exception as e:
        st.error(f"Error processing {file_path.name}: {str(e)}")
        return ""


# --------------------------
# Unified File Processor
# --------------------------
@st.cache_data(max_entries=5, ttl=3600)
def process_file(uploaded_file, temp_dir):
    file_path = Path(temp_dir) / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    
    if file_path.suffix.lower() == ".pdf":
        return process_pdf(file_path)
    elif file_path.suffix.lower() in (".pptx", ".ppt", ".doc", ".docx", ".xls", ".xlsx"):
        return process_office(file_path)
    elif file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        return extract_text_from_image(file_path.read_bytes())
    else:
        try:
            return textract.process(str(file_path)).decode("utf-8")
        except:
            return uploaded_file.getvalue().decode("utf-8", errors="replace")

# --------------------------
# Main Application
# --------------------------
def main():
    st.title("📄 Universal Document Processor")
    
    # Initialize session state
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents/Images",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing Files..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                    processed_content = list(executor.map(
                        lambda f: process_file(f, temp_dir),
                        uploaded_files
                    ))
                
                combined_content = "\n\n".join(processed_content)
                st.session_state.query_engine = VectorStoreIndex.from_documents(
                    [Document(text=combined_content)],
                    embed_model=embed_model
                ).as_query_engine(llm=groq_llm)

    # Chat interface
    if st.session_state.query_engine:
        st.header("💬 Document Insights")
        
        for msg in st.session_state.messages[-5:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Analyzing..."):
                response = st.session_state.query_engine.query(prompt)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.response
                })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response.response)

if __name__ == "__main__":
    main()
