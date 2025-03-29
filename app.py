import streamlit as st
import os
import tempfile
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import ImageNode
from llama_index.llms.groq import Groq
import fitz  # PyMuPDF
import pandas as pd
from pptx import Presentation
from docx import Document
import pytesseract
from PIL import Image
import io

# Initialize Groq LLM
groq_llm = Groq(model="llama-3.3-70b-specdec", api_key= st.secrets["k"]["api_key"])

# Multimodal Processing Functions
def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def process_pdf(file_path):
    """Extract text, images, and tables from PDF without saving images to disk"""
    full_text = ""
    doc = fitz.open(file_path)
    
    for page_num, page in enumerate(doc):
        # Extract text
        full_text += page.get_text()
        
        # Extract images
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            
            # Open the image in memory
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract text from the image using OCR
            extracted_text = pytesseract.image_to_string(image)
            full_text += f"\n[Image on page {page_num}, index {img_index}]: {extracted_text.strip()}"
            
    return full_text

def process_office_file(file_path, file_type):
    """Convert office files to images and extract text"""
    text_content = ""
    
    if file_type == "ppt":
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            img_path = f"temp_slide_{i}.png"
            slide.shapes.save(img_path)
            text_content += extract_text_from_image(img_path)
            
    elif file_type == "doc":
        doc = Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
        
    elif file_type == "xlsx":
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet_data in df.items():
            text_content += f"\nSheet: {sheet_name}\n"
            text_content += sheet_data.to_string()
    st.write(text_content)
    return text_content

# Document Processing Pipeline
def process_documents(uploaded_files):
    all_content = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext == ".pdf":
                content = process_pdf(file_path)
            elif file_ext in [".pptx", ".ppt"]:
                content = process_office_file(file_path, "ppt")
            elif file_ext in [".docx", ".doc"]:
                content = process_office_file(file_path, "doc")
            elif file_ext == ".xlsx":
                content = process_office_file(file_path, "xlsx")
            else:
                content = "Unsupported file type"
                
            all_content.append(content)
    
    return "\n\n".join(all_content)

# Vector Index Creation
def create_vector_index(content):
    documents = [Document(text=content)]
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(llm=groq_llm)

# Streamlit UI
st.title("🧠 Multimodal Document Analyzer")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOC, XLSX, PPT)",
    type=["pdf", "docx", "doc", "xlsx", "pptx", "ppt"],
    accept_multiple_files=True
)

# Processing Pipeline
if uploaded_files:
    with st.spinner("Processing documents..."):
        processed_content = process_documents(uploaded_files)
        query_engine = create_vector_index(processed_content)
        st.session_state.query_engine = query_engine
        st.success("Documents processed and indexed!")

# Chat Interface
if "query_engine" in st.session_state:
    st.header("Chat with Documents")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Generating response..."):
            response = st.session_state.query_engine.query(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})



# Setup Instructions
"""
1. Install Tesseract OCR:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: brew install tesseract
   - Linux: sudo apt install tesseract-ocr

2. Install dependencies:
   pip install -r requirements.txt

3. Set GROQ_API_KEY in Streamlit secrets
"""
