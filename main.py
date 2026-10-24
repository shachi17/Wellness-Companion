import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from memory import save_message, load_langchain_history

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



def get_medical_rag_response(user_question, vector_store):
    # Retrieve relevant passages from the indexed medical PDF
    documents = vector_store.similarity_search(
        user_question,
        k=3
    )

    if not documents:
        return "I couldn't find relevant information in the uploaded report."

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a medical report information assistant.

Answer the user's question using only the provided report excerpts.
Do not invent values or fill in missing information.
If the answer is not present in the excerpts, say you could not find it.
Preserve the reported values and units as written.
OCR text may contain errors, so warn the user to verify important values
against the original report.
Provide general educational information only; do not diagnose or prescribe.

Report excerpts:
{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = rag_prompt | llm_model | output_parser

    return rag_chain.invoke({
        "context": context,
        "question": user_question
    })



def get_medical_rag_response_with_memory(user_question, vector_store):
    # Load the previous conversation
    history = load_langchain_history()

    # Retrieve relevant information from the medical report
    documents = vector_store.similarity_search(
        user_question,
        k=3
    )

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Wellness Companion, a friendly medical report
            information assistant.

            Use the report excerpts to answer questions about report values.
            Use conversation history to understand follow-up questions.
            Never invent missing medical values.
            If the report does not contain the answer, say so.
            OCR may contain errors, so advise users to verify important
            values against the original report.
            Provide general educational information only.
            Do not diagnose or prescribe.

            Report excerpts:
            {context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    rag_chain = rag_prompt | llm_model | output_parser

    answer = rag_chain.invoke({
        "context": context,
        "chat_history": history,
        "question": user_question
    })

    # Save both sides of the conversation
    save_message("user", user_question)
    save_message("assistant", answer)

    return answer

