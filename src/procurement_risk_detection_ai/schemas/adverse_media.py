from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List


class AdverseMediaItem(BaseModel):
    entity: str = Field(..., description="Company or person name")
    allegation_type: str = Field(..., description="e.g., bribery, bid-rigging")
    date: Optional[str] = Field(None, description="ISO date if available")
    location: Optional[str] = None
    source_url: Optional[HttpUrl] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    snippet: Optional[str] = Field(default=None, description="Short evidence extract")


class AdverseMediaPayload(BaseModel):
    items: List[AdverseMediaItem]
