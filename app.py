import streamlit as st
from main import get_bot_response
from langchain.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Wellness Companion", page_icon="💪", layout="centered")

st.markdown("""
    <style>
    /* Clean, professional dark theme without neon glare */
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

    /* Subtle, clean button tiles */
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


col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("<div style='text-align: right; font-size: 40px; margin-top: 4px;'>💪</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color: #F8FAFC; font-size: 34px; margin-top: 0px; margin-bottom: 0px;'>Wellness Companion</h1>", unsafe_allow_html=True)

st.divider()


if "message_history" not in st.session_state:
    st.session_state.message_history = []


if not st.session_state.message_history:
    
    st.markdown("""
        <div style='background-color: #1E293B; padding: 14px 18px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 15px;'>
            <p style='color: #F8FAFC; font-size: 15px; font-weight: 600; margin-bottom: 2px;'>👋 Hi! I'm your Wellness Companion.</p>
            <p style='color: #94A3B8; font-size: 13px; margin: 0;'>I can help you understand simple health & wellness topics.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 14px; margin-bottom: 12px;'>✨ What would you like to learn today?</p>", unsafe_allow_html=True)
    
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("😴 Why is sleep important?", use_container_width=True):
            st.session_state.current_query = "Why is sleep important?"
        if st.button("🏃 Benefits of exercise?", use_container_width=True):
            st.session_state.current_query = "What are the benefits of exercise?"
    with c2:
        if st.button("⚖️ What is BMI?", use_container_width=True):
            st.session_state.current_query = "What is BMI?"
        if st.button("💧 Why drink enough water?", use_container_width=True):
            st.session_state.current_query = "Why should we drink enough water?"
            
    st.markdown("<br>", unsafe_allow_html=True)


prompt = st.chat_input("Your turn to ask questions, type something...")

if "current_query" in st.session_state:
    prompt = st.session_state.current_query
    del st.session_state.current_query

for msg in st.session_state.message_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar_icon = "👤" if role == "user" else "💪" 
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(msg.content)

if prompt:
    st.session_state.message_history.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        
    with st.chat_message("assistant", avatar="💪"):
        with st.spinner("Thinking..."):
            response_stream = get_bot_response(
                user_message=prompt,
                chat_history=st.session_state.message_history[:-1]
            )
            final_text = st.write_stream(response_stream)
    
    st.session_state.message_history.append(AIMessage(content=final_text))
    st.rerun()