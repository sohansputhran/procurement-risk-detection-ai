from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Procurement Risk Detection AI API", version="0.1.0")


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


def demo_score(
    amount: float, past_awards_count: int, is_sanctioned: bool, adverse_media_count: int
) -> float:
    # Heuristic demo only: normalize features into [0,1] then average with weights.
    amt_norm = min(amount / 1_000_000.0, 1.0)  # cap at 1M for demo
    awards_norm = min(past_awards_count / 10.0, 1.0)
    media_norm = min(adverse_media_count / 5.0, 1.0)
    sanction_norm = 1.0 if is_sanctioned else 0.0
    # weights: sanction>media>awards>amount (purely illustrative)
    score = (
        0.45 * sanction_norm + 0.25 * media_norm + 0.2 * awards_norm + 0.1 * amt_norm
    )
    return round(score, 4)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    s = demo_score(
        req.amount, req.past_awards_count, req.is_sanctioned, req.adverse_media_count
    )
    return ScoreResponse(risk_score=s, notes="demo-only heuristic")
