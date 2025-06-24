from sentence_transformers import SentenceTransformer
import streamlit as st
from . import doc_parser

model = SentenceTransformer('all-MiniLM-L6-v2')
def embedding(chunks):
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    return chunks, embeddings