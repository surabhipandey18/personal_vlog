from . import embeddings_engine
import streamlit as st
import faiss
import numpy as np
from . import doc_parser
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def faiss_search(query, embeddings, chunks, model):
    # Convert embeddings to numpy
    embeddings = np.array(embeddings).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Encode the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Create FAISS index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    # Search top 5 results
    D, I = index.search(query_embedding, 5)
    retrieved_chunks = [chunks[i] for i in I[0]]

    return retrieved_chunks

