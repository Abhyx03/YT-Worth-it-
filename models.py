from pydantic import BaseModel, field_validator
from typing import Optional


class AnalyseRequest(BaseModel):
    goal: str
    urls: list[str]

    @field_validator("goal")
    @classmethod
    def goal_length(cls, v):
        v = v.strip()
        if len(v) < 10:
            raise ValueError("Goal must be at least 10 characters")
        if len(v) > 300:
            raise ValueError("Goal must be under 300 characters")
        return v

    @field_validator("urls")
    @classmethod
    def urls_count(cls, v):
        if not 1 <= len(v) <= 2:
            raise ValueError("Provide 1 or 2 YouTube URLs")
        return v


class TimestampResult(BaseModel):
    seconds: int
    display: str
    link: str


class ChapterResult(BaseModel):
    title: str
    start_seconds: int
    display: str
    link: str


class VideoResult(BaseModel):
    video_id: str
    original_url: str
    relevance_score: int
    summary: str
    clickbait_rating: str
    clickbait_reasoning: str
    answer_timestamp: TimestampResult
    chapters: list[ChapterResult]


class AnalyseResponse(BaseModel):
    results: list[VideoResult]


class ErrorResponse(BaseModel):
    error: str
    message: str
    video_id: Optional[str] = None
