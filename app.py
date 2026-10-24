
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage

from main import (
    get_bot_response,
    get_medical_rag_response_with_memory
)

from rag import process_medical_pdf
from vector_store import create_vector_store, load_vector_store
from memory import load_messages, clear_messages


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Wellness Companion",
    page_icon="💪",
    layout="centered"
)


# ---------------- ORIGINAL DESIGN ----------------

st.markdown("""
    <style>
    .stApp {
        background-color: #121824;
        color: #E2E8F0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 600px;
    }

    .stButton>button {
        background-color: #1E293B;
        color: #94A3B8;
        border: 1px solid #334155;
        border-radius: 10px;
        font-weight: 500;
        padding: 12px;
        transition: all 0.2s ease;
        width: 100%;
    }

    .stButton>button:hover {
        background-color: #334155;
        color: #F8FAFC;
        border-color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------- LOAD VECTOR STORE ----------------

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception:
        st.session_state.vector_store = None


# ---------------- SIDEBAR: MODE SELECTION ----------------

with st.sidebar:
    st.title("💪 Wellness Companion")
    st.caption("Choose how you want to use your assistant.")

    mode = st.radio(
        "Chat mode",
        [
            "General Wellness Chat",
            "Medical Report Chat"
        ],
        key="chat_mode"
    )

    st.divider()

    if mode == "Medical Report Chat":
        st.subheader("🩺 Medical Documents")

        uploaded_file = st.file_uploader(
            "Upload a medical PDF",
            type=["pdf"],
            key="medical_pdf_upload"
        )

        if st.button(
            "Index Medical PDF",
            use_container_width=True
        ):
            if uploaded_file is None:
                st.warning("Please upload a PDF first.")
            else:
                try:
                    with st.spinner(
                        "Extracting text and building search index..."
                    ):
                        chunks = process_medical_pdf(uploaded_file)

                        st.session_state.vector_store = (
                            create_vector_store(chunks)
                        )

                        st.session_state.indexed_file = (
                            uploaded_file.name
                        )

                    st.success(
                        f"Successfully indexed {len(chunks)} text chunks."
                    )

                except Exception as e:
                    st.error(f"PDF indexing failed: {e}")

        if st.session_state.vector_store is not None:
            st.success("Medical document index is ready.")
        else:
            st.info("Upload and index a PDF to begin.")

        st.caption(
            "Indexing a new PDF replaces the previously indexed "
            "document in the existing vector store."
        )

        st.divider()

        if st.button(
            "🗑️ Clear Medical Conversation",
            use_container_width=True
        ):
            try:
                clear_messages()
                st.session_state.confirm_cleared = True
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear conversation: {e}")

        if st.session_state.get("confirm_cleared", False):
            st.success("Medical conversation history cleared.")
            st.session_state.confirm_cleared = False

        st.caption(
            "Clearing conversation history does not delete the indexed PDF."
        )


# ---------------- HEADER ----------------

col1, col2 = st.columns([1, 8])

with col1:
    st.markdown(
        "<div style='text-align:right;font-size:40px;margin-top:4px;'>"
        "💪</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<h1 style='color:#F8FAFC;font-size:34px;"
        "margin-top:0;margin-bottom:0;'>"
        "Wellness Companion</h1>",
        unsafe_allow_html=True
    )

st.divider()


# ==================================================
# MODE 1: ORIGINAL GENERAL WELLNESS CHAT
# ==================================================

if mode == "General Wellness Chat":

    st.caption("General health and wellness education.")

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # Original welcome screen and suggestions
    if not st.session_state.message_history:

        st.markdown("""
            <div style='background-color:#1E293B;padding:14px 18px;
            border-radius:12px;border:1px solid #334155;
            margin-bottom:15px;'>
                <p style='color:#F8FAFC;font-size:15px;
                font-weight:600;margin-bottom:2px;'>
                    👋 Hi! I'm your Wellness Companion.
                </p>
                <p style='color:#94A3B8;font-size:13px;margin:0;'>
                    I can help you understand simple health & wellness topics.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align:center;color:#94A3B8;"
            "font-size:14px;margin-bottom:12px;'>"
            "✨ What would you like to learn today?</p>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button("😴 Why is sleep important?"):
                st.session_state.current_query = (
                    "Why is sleep important?"
                )
                st.rerun()

            if st.button("🏃 Benefits of exercise?"):
                st.session_state.current_query = (
                    "What are the benefits of exercise?"
                )
                st.rerun()

        with c2:
            if st.button("⚖️ What is BMI?"):
                st.session_state.current_query = "What is BMI?"
                st.rerun()

            if st.button("💧 Why drink enough water?"):
                st.session_state.current_query = (
                    "Why should we drink enough water?"
                )
                st.rerun()

    # Display the original general chat history
    for msg in st.session_state.message_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        avatar_icon = "👤" if role == "user" else "💪"

        with st.chat_message(role, avatar=avatar_icon):
            st.markdown(msg.content)

    prompt = st.chat_input(
        "Your turn to ask questions, type something..."
    )

    if "current_query" in st.session_state:
        prompt = st.session_state.pop("current_query")

    # Generate a response only when a question is submitted
    if prompt:
        st.session_state.message_history.append(
            HumanMessage(content=prompt)
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="💪"):
            response_placeholder = st.empty()
            final_text = ""

            try:
                with st.spinner("Thinking..."):
                    response_stream = get_bot_response(
                        user_message=prompt,
                        chat_history=(
                            st.session_state.message_history[:-1]
                        )
                    )

                    for chunk in response_stream:
                        final_text += chunk
                        response_placeholder.markdown(final_text)

                st.session_state.message_history.append(
                    AIMessage(content=final_text)
                )

            except Exception as e:
                error_text = str(e).lower()

                if any(word in error_text for word in [
                    "quota",
                    "resource_exhausted",
                    "429",
                    "rate limit",
                    "too many requests"
                ]):
                    message = (
                        "Gemini's current usage limit has been reached. "
                        "Please wait for your quota to reset or check your "
                        "Google AI quota and billing settings."
                    )
                else:
                    message = (
                        "I'm temporarily unable to generate a response. "
                        "Please try again shortly."
                    )

                if final_text:
                    response_placeholder.markdown(final_text)
                    st.session_state.message_history.append(
                        AIMessage(content=final_text)
                    )
                else:
                    st.warning(message)


# ==================================================
# MODE 2: MEDICAL REPORT CHAT WITH PERSISTENT MEMORY
# ==================================================

elif mode == "Medical Report Chat":

    st.caption(
        "Ask questions about your uploaded medical report."
    )

    st.subheader("Medical Conversation")

    try:
        medical_messages = load_messages()

        if not medical_messages:
            st.info(
                "Your medical conversation will appear here. "
                "Upload and index a report, then ask a question."
            )
        else:
            for message in medical_messages:
                role = (
                    "assistant"
                    if message["role"] == "assistant"
                    else "user"
                )

                avatar_icon = "💪" if role == "assistant" else "👤"

                with st.chat_message(role, avatar=avatar_icon):
                    st.markdown(message["content"])

    except Exception as e:
        st.error(f"Could not load conversation history: {e}")

    question = st.chat_input(
        "Ask a question about your medical report...",
        key="medical_question"
    )

    if question:
        if st.session_state.vector_store is None:
            st.warning(
                "Please upload and index a medical PDF before asking "
                "questions."
            )
        else:
            try:
                with st.spinner(
                    "Finding relevant information and preparing an answer..."
                ):
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
