from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import time

from procurement_risk_detection_ai.app.api.provenance import log_provenance
from .batch import router as batch_router
from .health import router as health_router

app = FastAPI(title="Procurement Risk Detection AI API", version="0.1.0")

# Include other endpoint groups
app.include_router(batch_router)
app.include_router(health_router)


# ----- Models for /v1/score (single-record demo heuristic) -------------------
class ScoreRequest(BaseModel):
    amount: float = Field(ge=0, description="Contract award amount")
    past_awards_count: int = Field(
        ge=0, description="Historical awards to same supplier"
    )
    is_sanctioned: bool = Field(description="Whether supplier is on a sanctions list")
    adverse_media_count: int = Field(ge=0, description="Count of adverse media hits")


class ScoreResponse(BaseModel):
    risk_score: float
    notes: str
    # Keep this so FastAPI returns it in the response
    provenance_id: Optional[str] = None


# -----------------------------------------------------------------------------


def demo_score(
    amount: float,
    past_awards_count: int,
    is_sanctioned: bool,
    adverse_media_count: int,
) -> float:
    """
    Demo-only heuristic scoring:
    Normalizes basic inputs and combines with simple weights.
    Replace with ML baseline in the next milestone.
    """
    amt_norm = min(amount / 1_000_000.0, 1.0)  # cap at 1M for demo
    awards_norm = min(past_awards_count / 10.0, 1.0)
    media_norm = min(adverse_media_count / 5.0, 1.0)
    sanction_norm = 1.0 if is_sanctioned else 0.0
    score = (
        0.45 * sanction_norm + 0.25 * media_norm + 0.2 * awards_norm + 0.1 * amt_norm
    )
    return round(score, 4)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """
    Single-record scoring endpoint (demo heuristic).
    Adds per-request provenance logging and returns provenance_id.
    """
    started = time.time()
    try:
        s = demo_score(
            req.amount,
            req.past_awards_count,
            req.is_sanctioned,
            req.adverse_media_count,
        )
        prov_id = log_provenance(
            endpoint="/v1/score",
            payload=req.dict(),  # Pydantic v1/v2 compatible
            started_at=started,
            status="ok",
        )
        return ScoreResponse(
            risk_score=s, notes="demo-only heuristic", provenance_id=prov_id
        )

    except Exception as e:
        # Log failure (logger is fail-safe; won’t raise)
        log_provenance(
            endpoint="/v1/score",
            payload=req.dict(),
            started_at=started,
            status="error",
            error=str(e),
        )
        raise
