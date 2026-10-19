"""SQLite-хранилище эмбеддингов лиц (без фото)."""
import sqlite3

import numpy as np

DB_PATH = "people.db"


def _con():
    return sqlite3.connect(DB_PATH)


def init():
    with _con() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS people ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "name TEXT NOT NULL,"
            "embedding BLOB NOT NULL)"
        )


def add_person(name, embedding):
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    with _con() as con:
        cur = con.execute(
            "INSERT INTO people (name, embedding) VALUES (?, ?)", (name, blob)
        )
        return cur.lastrowid


def all_people():
    with _con() as con:
        rows = con.execute("SELECT id, name, embedding FROM people").fetchall()
    return [(i, n, np.frombuffer(b, dtype=np.float32)) for i, n, b in rows]


def next_guest_number():
    """Следующий свободный номер для «Гость #N»."""
    with _con() as con:
        rows = con.execute(
            "SELECT name FROM people WHERE name LIKE 'Гость #%'"
        ).fetchall()
    nums = [int(t) for (n,) in rows if (t := n.rsplit("#", 1)[-1]).isdigit()]
    return max(nums, default=0) + 1
