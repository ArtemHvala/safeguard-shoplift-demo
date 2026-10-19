"""Идентификация лица по базе (косинусная близость)."""
import db
from faces import cosine

THRESHOLD = 0.7


def identify(embedding, threshold=THRESHOLD):
    """Возвращает {id, name, sim} лучшего совпадения или None."""
    best = None
    best_sim = -1.0
    for pid, name, emb in db.all_people():
        sim = cosine(embedding, emb)
        if sim > best_sim:
            best_sim, best = sim, (pid, name)
    if best is not None and best_sim >= threshold:
        return {"id": best[0], "name": best[1], "sim": best_sim}
    return None


GUEST_PREFIX = "Гость #"


def identify_or_register(embedding, threshold=THRESHOLD):
    """Ищет совпадение; если нет — заводит нового «гостя» с номером."""
    res = identify(embedding, threshold)
    if res:
        return {**res, "new": False}
    number = db.next_guest_number()
    name = f"{GUEST_PREFIX}{number}"
    pid = db.add_person(name, embedding)
    return {"id": pid, "name": name, "sim": None, "new": True}


def display_name(result):
    """'Гость #3' -> '#3'; только что созданный -> 'неизвестный #3'."""
    name = result["name"]
    if name.startswith(GUEST_PREFIX):
        tail = name[len(GUEST_PREFIX):]
        return f"неизвестный #{tail}" if result.get("new") else f"#{tail}"
    return name
