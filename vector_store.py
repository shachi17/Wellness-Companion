
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Store the vector database locally in your project
VECTOR_DB_PATH = "medical_chroma_db"

# Convert text into embeddings locally
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



def create_vector_store(chunks):
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="medical_documents",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    # Remove previously indexed chunks
    existing = vector_store.get()

    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    # Add the newly extracted chunks
    if chunks:
        vector_store.add_documents(chunks)

    return vector_store



def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        return None

    embeddings = get_embeddings()

    return Chroma(
        collection_name="medical_documents",
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
