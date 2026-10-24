
import streamlit as st
from rag import process_medical_pdf

st.title("Medical PDF Reader")

uploaded_file = st.file_uploader(
    "Upload a medical PDF",
    type=["pdf"]
)

if uploaded_file:
    try:
        chunks = process_medical_pdf(uploaded_file)

        st.success(
            f"PDF processed successfully! "
            f"Created {len(chunks)} text chunks."
        )

        st.subheader("Preview extracted content")

        for i, chunk in enumerate(chunks[:3], start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk.page_content)
            st.caption(
                f"Source: {chunk.metadata.get('source')} | "
                f"Page: {chunk.metadata.get('page', 'Unknown')}"
            )

    except Exception as e:
        st.error(f"Could not process the PDF: {e}")
