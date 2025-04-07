import streamlit as st
from document_handler import load_and_chunk
from embed_store import load_embeddings, store_embeddings, retrieve_chunks
from rag_pipeline import generate_answer
from model_utils import load_embedding_model, load_llm_model

st.title("RAG-based Document Chatbot")

# Upload document
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
query = st.text_input("Ask a question about the document")

# Load models
embedding_model = load_embedding_model()
llm = load_llm_model()

if uploaded_file:
# app.py snippet update
    with st.spinner("Processing document..."):
        index, chunk_texts = load_embeddings()
        if not index:
            chunks = load_and_chunk(uploaded_file)
            index, chunk_texts = store_embeddings(chunks, embedding_model)
        st.success("Document ready for Q&A!")


    if query:
        with st.spinner("Generating answer..."):
            context = retrieve_chunks(query, embedding_model, index, chunk_texts)
            response = generate_answer(query, context, llm)
            with st.expander("See Answer"):
                st.markdown(response, unsafe_allow_html=True)


