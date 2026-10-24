
import streamlit as st

from rag import process_medical_pdf
from vector_store import create_vector_store

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
        results = st.session_state.vector_store.similarity_search(
            question,
            k=2
        )

        st.subheader("Relevant passages")

        for i, result in enumerate(results, start=1):
            st.markdown(f"**Passage {i}**")
            st.write(result.page_content)
            st.caption(
                f"Source: {result.metadata.get('source')} | "
                f"Page: {result.metadata.get('page', 'Unknown')}"
            )
