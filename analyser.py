import json

from fastapi import HTTPException
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    VideoUnavailable,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from google import genai
from google.genai import types


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetches transcript entries: [{"text": str, "start": float, "duration": float}, ...]
    Raises HTTPException on failure.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        transcript = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    except VideoUnavailable:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "VIDEO_NOT_FOUND",
                "message": "This video is unavailable or private.",
                "video_id": video_id,
            },
        )
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "NO_TRANSCRIPT",
                "message": (
                    "No transcript is available for this video. "
                    "It may be a music video, live stream, or have captions disabled."
                ),
                "video_id": video_id,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "TRANSCRIPT_FETCH_ERROR",
                "message": f"Could not fetch transcript: {str(e)}",
                "video_id": video_id,
            },
        )

    if len(transcript) < 30:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "TRANSCRIPT_TOO_SHORT",
                "message": "This video has too little spoken content to analyse meaningfully.",
                "video_id": video_id,
            },
        )
    return transcript


def format_transcript_for_prompt(entries: list[dict], max_chars: int = 54000) -> str:
    """
    Converts transcript entries to timestamped text for the prompt.
    Format: [M:SS / Xs] text
    - M:SS helps Gemini understand pacing
    - Xs is the raw integer seconds Gemini echoes back in JSON fields
    For long transcripts, keeps first 40% + last 10%.
    """
    lines = []
    for e in entries:
        s = int(e["start"])
        m, sec = divmod(s, 60)
        lines.append(f"[{m}:{sec:02d} / {s}s] {e['text']}")

    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    front = int(len(lines) * 0.40)
    back = int(len(lines) * 0.10)
    truncated = (
        lines[:front]
        + ["\n... [transcript truncated — middle section omitted for length] ...\n"]
        + lines[-back:]
    )
    return "\n".join(truncated)


ANALYSIS_PROMPT_TEMPLATE = """You are a helpful assistant that analyses YouTube video transcripts.

USER GOAL: {goal}

VIDEO ID: {video_id}

TRANSCRIPT (format: [M:SS / Xs] spoken text):
{transcript}

---

Analyse this transcript against the user's goal and respond with ONLY a valid JSON object matching this exact schema. Do not add any text outside the JSON.

{{
  "relevance_score": <integer 0-10, how well this video serves the user's stated goal>,
  "summary": "<exactly 2 sentences describing what the video covers>",
  "answer_timestamp_seconds": <integer, the Xs value from the transcript line where the core answer to the user's goal first appears; use 0 if content starts immediately>,
  "chapters": [
    {{
      "title": "<descriptive chapter name>",
      "start_seconds": <integer Xs value from the nearest transcript line>
    }}
  ]
}}

CHAPTERS GUIDANCE:
- Identify 3 to 8 logical topic segments based on content shifts in the transcript
- Use the raw integer second values (Xs) that appear in the transcript timestamps
- Chapters must be in ascending order of start_seconds
- First chapter must start at 0

RELEVANCE SCORE GUIDANCE:
- 9-10: Video directly and comprehensively answers the goal
- 7-8:  Video is largely relevant with minor gaps
- 5-6:  Partially relevant; covers related concepts but not the goal specifically
- 3-4:  Tangentially related; user would need other resources
- 0-2:  Not relevant to the goal"""


def analyse_video(video_id: str, goal: str, transcript_entries: list[dict], client: genai.Client) -> dict:
    """
    Returns parsed Gemini JSON with keys:
      relevance_score, summary, answer_timestamp_seconds, chapters
      answer_timestamp_seconds, chapters
    Raises HTTPException on API or parse failure.
    """
    transcript_text = format_transcript_for_prompt(transcript_entries)
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        goal=goal,
        video_id=video_id,
        transcript=transcript_text,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )
        raw_json = response.text.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1]
            raw_json = raw_json.rsplit("```", 1)[0].strip()
    except Exception as e:
        err_str = str(e)
        print(f"[Gemini ERROR] {err_str}")
        if "429" in err_str or "quota" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RATE_LIMIT",
                    "message": "Gemini API quota exceeded. Please wait a moment and try again.",
                    "video_id": video_id,
                },
            )
        raise HTTPException(
            status_code=502,
            detail={
                "error": "GEMINI_API_ERROR",
                "message": f"AI analysis failed: {err_str}",
                "video_id": video_id,
            },
        )

    try:
        data = json.loads(raw_json)
        required = [
            "relevance_score", "summary", "answer_timestamp_seconds", "chapters",
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing key: {key}")
        data["relevance_score"] = max(0, min(10, int(data["relevance_score"])))
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "GEMINI_PARSE_ERROR",
                "message": "AI returned an unexpected response format. Please try again.",
                "video_id": video_id,
            },
        )

    return data


COMPARE_PROMPT_TEMPLATE = """You are a helpful assistant comparing two YouTube videos for a user.

USER GOAL: {goal}

VIDEO A (id: {video_id_a}):
- Relevance score: {score_a}/10
- Summary: {summary_a}

VIDEO B (id: {video_id_b}):
- Relevance score: {score_b}/10
- Summary: {summary_b}

---

Based on the user's goal, decide which video is the better watch. Respond with ONLY a valid JSON object. Do not add any text outside the JSON.

{{
  "winner_video_id": "<video_id_a or video_id_b>",
  "reasoning": "<2-3 sentences explaining why this video is the better choice for the user's goal>"
}}"""


def compare_videos(
    goal: str,
    video_id_a: str,
    analysis_a: dict,
    video_id_b: str,
    analysis_b: dict,
    client: genai.Client,
) -> dict:
    """
    Compares two already-analysed videos and returns {winner_video_id, reasoning}.
    Raises HTTPException on failure.
    """
    prompt = COMPARE_PROMPT_TEMPLATE.format(
        goal=goal,
        video_id_a=video_id_a,
        score_a=analysis_a["relevance_score"],
        summary_a=analysis_a["summary"],
        video_id_b=video_id_b,
        score_b=analysis_b["relevance_score"],
        summary_b=analysis_b["summary"],
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
        raw_json = response.text.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[1]
            raw_json = raw_json.rsplit("```", 1)[0].strip()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "GEMINI_API_ERROR",
                "message": f"Comparison failed: {str(e)}",
                "video_id": None,
            },
        )

    try:
        data = json.loads(raw_json)
        if "winner_video_id" not in data or "reasoning" not in data:
            raise ValueError("Missing keys")
        if data["winner_video_id"] not in (video_id_a, video_id_b):
            data["winner_video_id"] = video_id_a if analysis_a["relevance_score"] >= analysis_b["relevance_score"] else video_id_b
    except (json.JSONDecodeError, ValueError, TypeError):
        raise HTTPException(
            status_code=500,
            detail={
                "error": "GEMINI_PARSE_ERROR",
                "message": "Comparison returned an unexpected format. Please try again.",
                "video_id": None,
            },
        )

    return data
