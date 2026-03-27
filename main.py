import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from models import AnalyseRequest
from utils import extract_video_id, make_timestamp_result, seconds_to_display, build_timestamp_link
from analyser import fetch_transcript, analyse_video

# Load API key from Gemini_API.env in the same directory as this file
env_path = Path(__file__).parent / "Gemini_API.env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Make sure Gemini_API.env contains: GEMINI_API_KEY=your_key"
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Worth It? YouTube Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyse")
async def analyse(request: AnalyseRequest):
    results = []

    for url in request.urls:
        video_id = extract_video_id(url)
        if not video_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_URL",
                    "message": f"'{url}' is not a valid YouTube URL.",
                    "video_id": None,
                },
            )

        transcript_entries = fetch_transcript(video_id)
        gemini_data = analyse_video(video_id, request.goal, transcript_entries, gemini_client)

        answer_ts = make_timestamp_result(video_id, gemini_data["answer_timestamp_seconds"])

        chapters = []
        for ch in gemini_data["chapters"]:
            s = int(ch["start_seconds"])
            chapters.append({
                "title": ch["title"],
                "start_seconds": s,
                "display": seconds_to_display(s),
                "link": build_timestamp_link(video_id, s),
            })

        results.append({
            "video_id": video_id,
            "original_url": url,
            "relevance_score": gemini_data["relevance_score"],
            "summary": gemini_data["summary"],
            "answer_timestamp": answer_ts,
            "chapters": chapters,
        })

    return {"results": results}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)
