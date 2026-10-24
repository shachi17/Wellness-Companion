
import streamlit as st

from rag import process_medical_pdf
from vector_store import create_vector_store, load_vector_store
from main import get_medical_rag_response_with_memory
from memory import load_messages, clear_messages


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Wellness Companion",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Wellness Companion")
st.caption(
    "Your medical report assistant — ask questions about your "
    "uploaded reports and revisit your conversation."
)


# ---------------- SESSION STATE ----------------

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception:
        st.session_state.vector_store = None

if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Medical Documents")

    uploaded_file = st.file_uploader(
        "Upload a medical PDF",
        type=["pdf"]
    )

    if st.button("Index Medical PDF", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        else:
            try:
                with st.spinner(
                    "Extracting text and building search index..."
                ):
                    chunks = process_medical_pdf(uploaded_file)
                    st.session_state.vector_store = create_vector_store(
                        chunks
                    )
                    st.session_state.indexed_file = uploaded_file.name

                st.success(
                    f"Successfully indexed {len(chunks)} text chunks."
                )

            except Exception as e:
                st.error(f"PDF indexing failed: {e}")

    st.divider()

    if st.session_state.vector_store is not None:
        st.success("Medical document index is ready.")
    else:
        st.info("Upload and index a PDF to begin.")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        try:
            clear_messages()
            st.session_state.confirm_cleared = True
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear conversation: {e}")

    if st.session_state.get("confirm_cleared", False):
        st.success("Conversation history cleared.")
        st.session_state.confirm_cleared = False

    st.caption(
        "Clearing the conversation does not delete the indexed documents."
    )


# ---------------- CHAT HISTORY ----------------

st.subheader("Conversation")

try:
    messages = load_messages()

    if not messages:
        st.info(
            "Your conversation will appear here. "
            "Try asking a question about your medical report."
        )
    else:
        for message in messages:
            role = (
                "assistant"
                if message["role"] == "assistant"
                else "user"
            )

            with st.chat_message(role):
                st.markdown(message["content"])

except Exception as e:
    st.error(f"Could not load conversation history: {e}")


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask a question about your medical report..."
)

if question:
    if st.session_state.vector_store is None:
        st.warning(
            "Please upload and index a medical PDF before asking questions."
        )
    else:
        try:
            with st.spinner("Finding relevant information and preparing an answer..."):
                answer = get_medical_rag_response_with_memory(
                    question,
                    st.session_state.vector_store
                )

            st.rerun()

        except Exception as e:
            st.error(
                "I couldn't generate an answer right now. "
                "Please try again."
            )
            st.caption(f"Error details: {e}")
