import sqlite3
import os
import json

DB_NAME = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            filename TEXT,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    ''')

    # Migrate: add columns if they don't exist yet
    for col in [("page_count", "INTEGER DEFAULT 0"),
                ("chunk_count", "INTEGER DEFAULT 0"),
                ("image_count", "INTEGER DEFAULT 0")]:
        try:
            cursor.execute(f"ALTER TABLE sessions ADD COLUMN {col[0]} {col[1]}")
        except Exception:
            pass

    try:
        cursor.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
    except Exception:
        pass

    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_session(title, filename=None, page_count=0, chunk_count=0, image_count=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO sessions (title, filename, page_count, chunk_count, image_count) VALUES (?, ?, ?, ?, ?)',
        (title, filename, page_count, chunk_count, image_count)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def update_session_stats(session_id, page_count, chunk_count, image_count):
    conn = get_db_connection()
    conn.execute(
        'UPDATE sessions SET page_count = ?, chunk_count = ?, image_count = ? WHERE id = ?',
        (page_count, chunk_count, image_count, session_id)
    )
    conn.commit()
    conn.close()


def add_message(session_id, sender, content, metadata=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    meta_json = json.dumps(metadata) if metadata else None
    cursor.execute(
        'INSERT INTO messages (session_id, sender, content, metadata) VALUES (?, ?, ?, ?)',
        (session_id, sender, content, meta_json)
    )
    conn.commit()
    conn.close()


def get_all_sessions():
    conn = get_db_connection()
    sessions = conn.execute(
        'SELECT * FROM sessions ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return [dict(s) for s in sessions]


def get_session_messages(session_id, limit=50):
    conn = get_db_connection()
    messages = conn.execute(
        'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?',
        (session_id, limit)
    ).fetchall()
    conn.close()
    result = []
    for m in messages:
        row = dict(m)
        if row.get('metadata'):
            try:
                row['metadata'] = json.loads(row['metadata'])
            except Exception:
                row['metadata'] = None
        result.append(row)
    return result


def get_session_info(session_id):
    conn = get_db_connection()
    session = conn.execute(
        'SELECT * FROM sessions WHERE id = ?', (session_id,)
    ).fetchone()
    conn.close()
    return dict(session) if session else None


def rename_session(session_id, title):
    conn = get_db_connection()
    conn.execute('UPDATE sessions SET title = ? WHERE id = ?', (title, session_id))
    conn.commit()
    conn.close()


def delete_session(session_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()


def get_recent_messages(session_id, n=6):
    """Return last n messages for conversation context."""
    conn = get_db_connection()
    messages = conn.execute(
        'SELECT sender, content FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?',
        (session_id, n)
    ).fetchall()
    conn.close()
    return list(reversed([dict(m) for m in messages]))
