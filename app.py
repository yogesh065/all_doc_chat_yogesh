import streamlit as st
import os
import tempfile
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq
import fitz  # PyMuPDF
import pandas as pd
from pptx import Presentation
from llama_index.core import Document
import pytesseract
from PIL import Image
import io
import pytesseract
import os
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
# Set up the HuggingFaceEmbedding class with the required model to use with llamaindex core.
embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-small-en")
Settings.embed_model = embed_model
# Initialize Groq LLM
groq_llm = Groq(model="llama-3.3-70b-specdec", api_key= st.secrets["k"]["api_key"])
import os
import pytesseract

import pytesseract
import shutil

tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise EnvironmentError('Tesseract not found in system PATH')

print(f"Tesseract path set to: {pytesseract.pytesseract.tesseract_cmd}")
def extract_text_from_image(image_data):
    """Extract text from image using Tesseract OCR"""
    try:
        # Open the image from in-memory data or file path
        if isinstance(image_data, str):  # If a file path is provided
            img = Image.open(image_data)
        else:  # If in-memory bytes are provided
            img = Image.open(io.BytesIO(image_data))
        
        # Perform OCR using Tesseract
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
            for shape in slide.shapes:
                # Extract text from text frames
                if shape.has_text_frame:
                    text_content += shape.text_frame.text + "\n"
                
                # Extract images from slides
                if shape.shape_type == 13:  # Shape type 13 corresponds to pictures
                    image = shape.image
                    image_bytes = image.blob
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # Perform OCR on the image
                    extracted_text = extract_text_from_image(io.BytesIO(image_bytes))
                    text_content += f"\n[Image on slide {i}]: {extracted_text.strip()}"
                    
    elif file_type == "doc":
        doc = Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "xlsx":
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet_data in df.items():
            text_content += f"\nSheet: {sheet_name}\n"
            text_content += sheet_data.to_string()
    elif file_type == "docx":
        doc = Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    elif file_type in (("jpg", "jpeg", "png", "gif")):
        image = Image.open(file_path)
        text_content = extract_text_from_image(image)
    st.write(text_content)

    return text_content

def process_documents(uploaded_files):
    """Process uploaded documents and extract their content"""
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
def create_vector_index(content):
    """Create a vector index for querying document content using BGE embeddings"""
    # Initialize BGE embedding model with proper configuration
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en"
    )
    
    # Configure text splitting parameters
    Settings.text_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Create document and index
    document = Document(text=content)
    index = VectorStoreIndex.from_documents(
        [document],
        embed_model=embed_model,  # Explicitly pass configured embed_model
        show_progress=True
    )
    
    return index.as_query_engine(llm=groq_llm)

# Streamlit UI
st.title("🧠 Multimodal Document Analyzer")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOC, XLSX, PPT)",
    type=["pdf", "docx", "doc", "xlsx", "pptx", "ppt","png","jpg"],
    accept_multiple_files=False
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

