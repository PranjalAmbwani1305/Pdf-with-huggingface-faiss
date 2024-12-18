import os
import PyPDF2
import torch
import faiss
import numpy as np
import streamlit as st
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModel

def create_pdf(filename="example.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello, this is a sample PDF document.", ln=True, align='C')
    pdf.output(filename)
    return filename

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading Hugging Face model: {e}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def create_faiss_index(embedding_dimension):
    index = faiss.IndexFlatL2(embedding_dimension)
    return index

def store_embeddings_in_faiss(embeddings, index):
    embeddings = np.array([embeddings], dtype=np.float32)
    index.add(embeddings)
    st.write("Embedding stored in FAISS.")

def query_faiss_index(query_text, index):
    query_embedding = generate_embeddings(query_text)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    D, I = index.search(query_embedding, k=1)
    return I, D

def main():
    st.title("PDF Similarity Search with FAISS and Hugging Face")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_path = f"uploaded_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_text = extract_text_from_pdf(pdf_path)

        embeddings = generate_embeddings(pdf_text)

        index = create_faiss_index(len(embeddings))
        store_embeddings_in_faiss(embeddings, index)

        st.subheader("PDF Content:")
        st.text(pdf_text)

    query = st.text_input("Enter a query to find the most similar PDF content:")

    if query:
        if uploaded_file is not None:
            I, D = query_faiss_index(query, index)
            st.write(f"Most similar PDF (index {I})")
            st.write(f"Similarity distance: {D}")
        else:
            st.write("Please upload a PDF first.")

   
if __name__ == "__main__":
    main()
