
import sqlite3
from pathlib import Path

# Store conversation history in the project folder
DB_PATH = Path(__file__).parent / "conversation_memory.db"


def init_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_message(role, content):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content)
        )


def load_messages():
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def clear_messages():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM messages")


def load_langchain_history():
    messages = load_messages()

    return [
        ("human" if message["role"] == "user" else "ai",
         message["content"])
        for message in messages
    ]


init_db()
