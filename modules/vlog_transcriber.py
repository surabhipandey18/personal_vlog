import os
import streamlit as st


def vlog_entry(vlog_path):
    st.subheader("Vlog Gallery")
    for file in os.listdir(vlog_path):
        if file.endswith(('.mp4', '.mov', '.avi')):
            st.video(os.path.join(vlog_path, file))
