import fitz
import streamlit as st
import re
import docx
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='docx')
from docx import Document 

def clean_text(text):
    # Replace all whitespace (including newlines, tabs) with a single space
    return re.sub(r'\s+', ' ', text).strip()

def doc_parser(file):
    if file is not None:
        st.write("File uploaded successfully!")
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            #clean up the text extraction    
            return clean_text(text)

        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file.seek(0)  # Reset file pointer to the beginning
            doc = Document(file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            #Clean up the text extraction
            return clean_text(text)

        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            #Clean up the text extraction
            return clean_text(text)
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
            return None

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks