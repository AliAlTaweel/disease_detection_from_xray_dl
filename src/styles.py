import streamlit as st

def apply_styles():
    st.set_page_config(page_title="Pneumonia AI Diagnostic", page_icon="🫁")
    st.title("🫁 Chest X-Ray Diagnostic")
    st.markdown("Upload a patient's chest X-ray to detect signs of **Pneumonia**.")

DISCLAIMER = "**Disclaimer:** This is an AI-assisted tool and should not replace a professional medical diagnosis."