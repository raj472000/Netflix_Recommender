from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routes import router
from app.middleware.request_middleware import RequestMiddleware


app = FastAPI(
    title="Advanced Recommender V2",
    version="2.0.0",
)

#app.add_middleware(RequestMiddleware)

app.include_router(router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)