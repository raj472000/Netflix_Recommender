import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ERROR_COUNT,
)


class RequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        endpoint = request.url.path
        method = request.method

        try:
            response: Response = await call_next(request)

            latency = time.time() - start_time

            REQUEST_COUNT.labels(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
            ).inc()

            REQUEST_LATENCY.labels(
                endpoint=endpoint,
                method=method,
            ).observe(latency)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(latency, 4))

            return response

        except Exception:
            ERROR_COUNT.labels(
                endpoint=endpoint,
                method=method,
            ).inc()
            raise