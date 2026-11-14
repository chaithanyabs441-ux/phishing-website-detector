"""Request schemas for the web application's JSON endpoints."""

from pydantic import BaseModel, Field


class ScanPayload(BaseModel):
    url: str = Field(..., max_length=4096)
    html: str = Field(default="", max_length=12000000)
