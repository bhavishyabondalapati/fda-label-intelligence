"""Wait until all four services answer /health (or just print their state with --once)."""
import sys
import time

import httpx

PORTS = {"ingest": 8001, "chunk": 8002, "embed": 8003, "query": 8004}
if "--answer" in sys.argv:          # Phase 2: wait only for the answer service
    PORTS = {"answer": 8005}


def check():
    state = {}
    for name, port in PORTS.items():
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            state[name] = r.json() if r.status_code == 200 else f"HTTP {r.status_code}"
        except httpx.HTTPError:
            state[name] = None
    return state


if __name__ == "__main__":
    once = "--once" in sys.argv
    deadline = time.time() + 300         # the embed service needs a few seconds to load the model
    while True:
        state = check()
        if once or all(state.values()) or time.time() > deadline:
            break
        time.sleep(1)
    for name, st in state.items():
        print(f"  {name:7s} :{PORTS[name]}  {'UP   ' + str(st) if st else 'DOWN'}")
    sys.exit(0 if all(state.values()) else 1)
