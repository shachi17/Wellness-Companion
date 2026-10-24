
import pymupdf
import easyocr
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def process_medical_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    reader = get_ocr_reader()
    documents = []

    try:
        for page_number, page in enumerate(pdf):
            # Extract existing PDF text
            extracted_text = page.get_text().strip()

            # Run OCR on the page image as well
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )

            image_bytes = pix.tobytes("png")

            results = reader.readtext(
                image_bytes,
                detail=0
            )

            ocr_text = "\n".join(results).strip()

            # Display both results for debugging
            with st.expander(
                f"Debug: Page {page_number + 1}"
            ):
                st.write("Text extracted directly from PDF:")
                st.code(extracted_text or "No text found")

                st.write("Text detected by OCR:")
                st.write(results or "No text detected")

            # Use whichever result contains more text
            if len(ocr_text) > len(extracted_text):
                text = ocr_text
            else:
                text = extracted_text

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number
                        }
                    )
                )

    finally:
        pdf.close()

    if not documents:
        raise ValueError(
            "No readable text found in this PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    return splitter.split_documents(documents)
