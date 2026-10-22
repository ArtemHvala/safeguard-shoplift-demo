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
        con.execute(
            "CREATE TABLE IF NOT EXISTS cam_sightings ("
            "person_id INTEGER NOT NULL,"
            "cam INTEGER NOT NULL,"
            "ts REAL NOT NULL,"
            "PRIMARY KEY (person_id, cam))"
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


def note_sighting(person_id, cam, ts):
    """Запоминает ПЕРВОЕ появление человека на камере cam (1 или 2)."""
    with _con() as con:
        con.execute(
            "INSERT OR IGNORE INTO cam_sightings (person_id, cam, ts)"
            " VALUES (?, ?, ?)",
            (int(person_id), int(cam), float(ts)),
        )


def cam_timeline():
    """[(person_id, name, t1, t2)] — первое появление на камере 1 и на камере 2."""
    with _con() as con:
        rows = con.execute(
            "SELECT p.id, p.name,"
            " MAX(CASE WHEN s.cam = 1 THEN s.ts END) AS t1,"
            " MAX(CASE WHEN s.cam = 2 THEN s.ts END) AS t2"
            " FROM people p"
            " LEFT JOIN cam_sightings s ON s.person_id = p.id"
            " GROUP BY p.id, p.name"
            " ORDER BY p.id"
        ).fetchall()
    return rows


def reset_sightings():
    """Сбрасывает времена появлений (сами люди остаются в базе)."""
    with _con() as con:
        con.execute("DELETE FROM cam_sightings")
