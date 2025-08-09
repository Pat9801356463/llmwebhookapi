import uuid
from collections import deque
from typing import Dict, List, Optional
from engine.lru_cache import session_cache  # Shared LRU store
from config import Config

# Max history stored per session
MAX_HISTORY = Config.CACHE_MAX_QUERIES if hasattr(Config, "CACHE_MAX_QUERIES") else 5

def start_session() -> str:
    """Starts a new session and returns its UUID-based ID"""
    session_id = str(uuid.uuid4())
    session_cache.set(session_id, deque(maxlen=MAX_HISTORY))
    return session_id

def get_session_history(session_id: str) -> List[Dict]:
    """Returns the query/response history for a session"""
    history = session_cache.get(session_id)
    return list(history) if history else []

def add_to_session(session_id: str, user_query: str, system_response: Dict):
    """Adds an interaction to session history"""
    history = session_cache.get(session_id)

    if history is None:
        history = deque(maxlen=MAX_HISTORY)

    history.append({
        "query": user_query,
        "response": system_response
    })

    session_cache.set(session_id, history)

def get_last_decision(session_id: str) -> Optional[Dict]:
    """Returns the last system response from session"""
    history = session_cache.get(session_id)
    if history and len(history) > 0:
        return history[-1].get("response")
    return None

# === Alias for compatibility with app.py ===
get_session_context = get_session_history
update_session = add_to_session  # ✅ alias for consistency

# 🔬 CLI Test
if __name__ == "__main__":
    sid = start_session()
    print("🔑 Session ID:", sid)

    q = "46M, knee surgery, 3-month policy"
    r = {"decision": "approved", "amount": 40000, "justification": "Clause 12 applies"}

    add_to_session(sid, q, r)

    print("📜 Session History:", get_session_history(sid))
    print("🔁 Last Decision:", get_last_decision(sid))
