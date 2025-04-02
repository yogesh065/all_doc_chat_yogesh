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
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
import shutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure HuggingFace Embedding Model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

# Initialize Groq LLM
groq_llm = Groq(model="llama-3.3-70b-specdec", api_key=st.secrets["k"]["api_key"])

# Configure Tesseract OCR Path
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise EnvironmentError('Tesseract not found in system PATH')

# Function to Extract Text from Images Using OCR
def extract_text_from_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)) if isinstance(image_data, bytes) else Image.open(image_data)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

# Function to Process PDF Files
def process_pdf(file_path):
    full_text = ""
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        full_text += page.get_text()
        for img_index, img in enumerate(page.get_images()):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            extracted_text = extract_text_from_image(image_bytes)
            full_text += f"\n[Image on page {page_num}, index {img_index}]: {extracted_text}"
    return full_text

# Function to Process Office Files (PPT, DOC, XLSX)
def process_office_file(file_path, file_type):
    text_content = ""
    if file_type == "ppt":
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text_content += shape.text_frame.text + "\n"
                if shape.shape_type == 13:  # Picture type
                    image_bytes = shape.image.blob
                    extracted_text = extract_text_from_image(image_bytes)
                    text_content += f"\n[Image on slide {i}]: {extracted_text}"
    elif file_type == "doc":
        doc = Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "xlsx":
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet_data in df.items():
            text_content += f"\nSheet: {sheet_name}\n{sheet_data.to_string()}"
    return text_content

# Function to Process Uploaded Documents
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

# Function to Create Vector Index for Querying Documents
def create_vector_index(content):
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    document = Document(text=content)
    index = VectorStoreIndex.from_documents([document], embed_model=embed_model, show_progress=True)
    return index.as_query_engine(llm=groq_llm)

# Streamlit UI Setup with Session Management and Chat Interface
st.title("🧠 Multimodal Document Analyzer")

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, XLSX, PPTX)",
    type=["pdf", "docx", "doc", "xlsx", "pptx", "ppt"],
    accept_multiple_files=True,
)

image_input = st.file_uploader("Upload an Image for OCR", type=["png", "jpg", "jpeg"])

if uploaded_files or image_input:
    with st.spinner("Processing..."):
        if uploaded_files:
            processed_content = process_documents(uploaded_files)
            query_engine = create_vector_index(processed_content)
            st.session_state.query_engine = query_engine

        if image_input:
            image_bytes = image_input.read()
            ocr_text = extract_text_from_image(image_bytes)
            query_engine = create_vector_index(ocr_text)
            st.session_state.query_engine = query_engine
            
         

if st.session_state.query_engine:
    st.header("Chat with Your Documents")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response..."):
            response = st.session_state.query_engine.query(prompt)

        with st.chat_message("assistant"):
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
