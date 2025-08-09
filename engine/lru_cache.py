from collections import OrderedDict
from config import Config
import time

class LRUCache:
    def __init__(self, max_size=Config.CACHE_MAX_QUERIES, ttl_minutes=Config.SESSION_TTL_MINUTES):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_minutes * 60  # convert to seconds

    def _is_expired(self, entry):
        return time.time() - entry["timestamp"] > self.ttl

    def get(self, key):
        if key in self.cache:
            entry = self.cache.pop(key)
            if not self._is_expired(entry):
                # Move to end to show recent use
                self.cache[key] = entry
                return entry["value"]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove LRU
        self.cache[key] = {"value": value, "timestamp": time.time()}

    def clear(self):
        self.cache.clear()

    def get_all_keys(self):
        return list(self.cache.keys())

# Global shared instance
session_cache = LRUCache()
