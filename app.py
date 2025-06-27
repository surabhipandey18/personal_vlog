import streamlit as st
from modules import doc_parser, embeddings_engine, faiss_search, summarizer
import os
import datetime
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2') 

st.set_page_config(page_title = "Personal Vlogging-Diary-Document App", layout="wide")
st.title("Personal Vlogging-Blogging-Diary App")
st.sidebar.title("Navigation")
options = ["Vlog", "Diary", "Documents"]
selected_option = st.sidebar.selectbox("Watcha' upto today?", options)

if selected_option == "Vlog":
    video_file = st.file_uploader("Upload your vlog video", type=["mp4", "mov", "avi"])
    title = st.text_input("Enter the title of your vlog")
    description = st.text_area("Enter a description for your vlog", max_chars=500)
    
    vlog_path = "data/raw/vlogs/"
    os.makedirs(vlog_path, exist_ok=True)

    if video_file is not None and title and description:
        save_path = os.path.join(vlog_path, video_file.name)
        with open(save_path, "wb") as f:
            f.write(video_file.getbuffer())

        metadata_path = save_path + ".txt"
        with open(metadata_path, "w") as f:
            f.write(f"{title}\n{description}")
        st.success("File and metadata saved!")
    else:
        st.warning("Please upload a vlog video, title, and description.")

    st.subheader("Vlog Gallery")
    if len(os.listdir(vlog_path)) == 0:
        st.write("No vlogs found.")
    else:
        for file in sorted(os.listdir(vlog_path), reverse=True):
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                video_file_path = os.path.join(vlog_path, file)
                metadata_path = video_file_path + ".txt"

                with st.expander(f"🎥 {file}", expanded=False):
                    st.video(video_file_path)
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            lines = f.readlines()
                            st.subheader(lines[0].strip() if len(lines) > 0 else file)
                            st.caption(lines[1].strip() if len(lines) > 1 else "No description available.")
                    else:
                        st.subheader(file)
                        st.caption("No description available.")

                    if st.button(f"🗑️ Delete '{file}'", key=f"delete_vlog_{file}"):
                        os.remove(video_file_path)
                        if os.path.exists(metadata_path):
                            os.remove(metadata_path)
                        st.success(f"Deleted vlog '{file}' and its metadata.")
                        st.experimental_rerun()

elif selected_option == "Diary":
    st.subheader("Write your diary entry")
    st.title("How's everything going lately? Write it down here!")
    diary_text = st.text_area("Write your diary entry here", height=300)
    diary_path = "data/raw/diary/"
    os.makedirs(diary_path, exist_ok=True)
    if st.button("Save Diary Entry"):
        if diary_text.strip():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diary_entry_{timestamp}.txt"
            with open(os.path.join(diary_path, filename), "w") as f:
                f.text_area(diary_text)
            st.success(f"Diary entry saved as {filename}")
            # If you want to process or analyze the diary entry, call your function/module here.
            # For example, if you have a diary module with a diary function:
            # diary.diary(diary_text)
            pass  # No action needed unless you want to add extra processing
        else:
            st.warning("Please write something in your diary.")
    st.subheader("Diary Gallery")
    diary_files = [f for f in os.listdir(diary_path) if f.endswith(".txt")]
    if not diary_files:
        st.write("No diary entries found.")
    else:
        for file in sorted(diary_files, reverse=True):
            file_path = os.path.join(diary_path, file)
            with st.expander(f"📖 {file}", expanded=False):
                with open(file_path, "r") as f:
                    st.text(f.read())

                if st.button(f"🗑️ Delete '{file}'", key=f"delete_diary_{file}"):
                    os.remove(file_path)
                    st.success(f"Deleted diary entry: {file}")
                    st.experimental_rerun()
           
elif selected_option == "Documents":
    st.subheader("Upload and Parse Documents")
    file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    file_path = "data/raw/documents/"
    os.makedirs(file_path, exist_ok=True)

    if file is not None:
        doc_path = os.path.join(file_path, file.name)
        with open(doc_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Saved file: {doc_path}")

        text = doc_parser.doc_parser(file)
        if text:
            st.write("Document parsed successfully!")
            st.text_area("Parsed Text", text, height=300)

            chunks = doc_parser.chunk_text(text)
            st.write("Text chunks created successfully!")

            chunks, embeddings = embeddings_engine.embedding(chunks)
            st.write("Embeddings created successfully!")
            #let's summarize the document
            if st.button("Summarize Document"):
                summary_text = summarizer.safe_summarizer(" ".join(chunks))
                st.subheader("Summary of the Relevant Chunks")
                st.text_area(summary_text)
            # Search functionality
            query = st.text_input("Enter your search query")
            if query:
                if text:
                    retrieved_chunks = faiss_search.faiss_search(query, embeddings, chunks, model=model)
                    st.subheader("Search Results")
                    for chunk in retrieved_chunks:
                        st.text(chunk)
                else:
                    st.warning("No text available to search.")##
else:
    st.warning("Please upload a document.")
st.sidebar.subheader("About")
st.sidebar.write("This app allows you to upload and manage your vlogs, diaries, and documents. You can parse text, create embeddings, search for information, and generate summaries.")
st.sidebar.write("Developed by Surabhi Pandey.")            
    
