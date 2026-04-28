from fastapi import APIRouter, HTTPException, Query
from app.services.recommender import Recommender


router = APIRouter()
service = Recommender()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "recommender-v2",
    }


@router.get("/recommend")
def recommend(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        return service.recommend(query=query, top_k=top_k)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(exc)}",
        ) from exc