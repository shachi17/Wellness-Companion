import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm_model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    streaming=True
)

output_parser = StrOutputParser()


system_prompt = """You are Wellness Companion, a friendly AI for general health
and wellness education.

Answer the user's main question directly in simple, concise language.
Use short paragraphs or bullets when useful. Be empathetic and non-judgmental.

Provide general educational information only. Never diagnose conditions,
prescribe medication, or claim to replace a healthcare professional.
For serious or potentially life-threatening symptoms, recommend appropriate
professional or emergency care. Do not invent medical facts."""

def get_bot_response(user_message, chat_history):
     prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), 
     ("human", "{user_message}")
    ])

     chain = prompt_template | llm_model | output_parser
    
     return chain.stream({
    "chat_history": chat_history,
     "user_message": user_message
})
