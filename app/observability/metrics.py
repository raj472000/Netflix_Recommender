from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "recommender_request_count",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "recommender_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
)

ERROR_COUNT = Counter(
    "recommender_error_count",
    "Total number of errors",
    ["endpoint", "method"],
)

CACHE_HIT_COUNT = Counter(
    "recommender_cache_hit_count",
    "Total cache hits",
)

CACHE_MISS_COUNT = Counter(
    "recommender_cache_miss_count",
    "Total cache misses",
)

RETRIEVAL_COUNT = Counter(
    "recommender_retrieval_count",
    "Total retrieval calls",
)