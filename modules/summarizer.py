import textwrap
from transformers import pipeline, AutoTokenizer
from . import doc_parser
from . import embeddings_engine
from . import faiss_search
from sentence_transformers import SentenceTransformer

# Sentence embedding model for retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Summarization model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def safe_summarizer(text, max_length=150, min_length=50, do_sample=False):
    from transformers import pipeline
    import torch

    max_tokens = tokenizer.model_max_length  # 1024 for bart-large-cnn
    inputs = tokenizer(text, return_tensors="pt", truncation=False)

    # Get token count
    num_tokens = inputs["input_ids"].shape[1]

    if num_tokens <= max_tokens:
        # Text is within limit, summarize directly
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']

    else:
        # Break text into smaller token-aware chunks
        sentences = text.split('. ')
        current_chunk = ""
        chunks = []
        current_tokens = 0

        for sentence in sentences:
            token_count = len(tokenizer.tokenize(sentence))
            if current_tokens + token_count <= max_tokens:
                current_chunk += sentence + ". "
                current_tokens += token_count
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_tokens = token_count

        if current_chunk:
            chunks.append(current_chunk.strip())

        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(f"[Chunk {i+1} skipped due to error: {str(e)}]")

        return " ".join(summaries)
