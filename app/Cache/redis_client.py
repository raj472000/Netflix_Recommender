import json
import os
import redis
from typing import Any, Optional


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))


class RedisCache:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )

    def get(self, key: str) -> Optional[Any]:
        value = self.client.get(key)

        if value is None:
            return None

        return json.loads(value)

    def set(self, key: str, value: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
        self.client.setex(
            key,
            ttl,
            json.dumps(value),
        )