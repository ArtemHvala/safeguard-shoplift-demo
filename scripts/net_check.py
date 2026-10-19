"""Проверка доступности modal.com: прокси и напрямую."""
import time
import urllib.request

URL = "https://modal.com"
PROXY = "http://127.0.0.1:10809"


def check(label, opener):
    t = time.time()
    try:
        r = opener.open(URL, timeout=15)
        print(label, "->", r.status, f"{time.time() - t:.1f}s")
    except Exception as e:
        print(label, "-> ERR", type(e).__name__, str(e)[:150])


check("as-is (system proxy)", urllib.request.build_opener())
check("direct (no proxy)", urllib.request.build_opener(urllib.request.ProxyHandler({})))
check("via 127.0.0.1:10809", urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})))
