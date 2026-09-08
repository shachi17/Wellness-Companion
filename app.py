import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from main import get_bot_response

st.set_page_config(
    page_title="AI Health Companion", 
    page_icon="🩺", 
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 AI Health Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Your personalized wellness and nutrition guide</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar_icon = "👤" if role == "user" else "🏥" 
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(msg.content)

if chat_box := st.chat_input("How can I assist you?"):
    st.session_state.chat_history.append(HumanMessage(content=chat_box))
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(chat_box)
        
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Thinking..."):
            stream_generator = get_bot_response(
                user_message=chat_box,
                chat_history=st.session_state.chat_history[:-1]
            )
            final_result = st.write_stream(stream_generator)
    
    st.session_state.chat_history.append(AIMessage(content=final_result))
    st.rerun()