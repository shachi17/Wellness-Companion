
import streamlit as st

from rag import process_medical_pdf
from vector_store import create_vector_store
from main import get_medical_rag_response_with_memory

st.title("Medical RAG — Document Indexing")

uploaded_file = st.file_uploader(
    "Upload a medical PDF",
    type=["pdf"]
)

# Keep the indexed database available across Streamlit reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.button("Index Medical PDF") and uploaded_file:
    try:
        with st.spinner("Extracting text and building search index..."):
            chunks = process_medical_pdf(uploaded_file)
            st.session_state.vector_store = create_vector_store(chunks)

        st.session_state.indexed_file = uploaded_file.name
        st.success(f"Indexed {len(chunks)} text chunks.")

    except Exception as e:
        st.error(f"Indexing failed: {e}")

# This section is OUTSIDE the indexing button
if st.session_state.vector_store is not None:
    st.success(
        f"Ready to search: "
        f"{st.session_state.get('indexed_file', 'Medical PDF')}"
    )

    question = st.text_input(
        "Ask a question about this document",
        key="medical_question"
    )

    
    if question.strip():
        if st.button("Get Answer"):
            try:
                with st.spinner("Reading the report and generating an answer..."):
                    answer = get_medical_rag_response_with_memory(
                        question,
                        st.session_state.vector_store
                    )

                st.subheader("Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Could not generate an answer: {e}")
