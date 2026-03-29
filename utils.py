import urllib.parse
from typing import Optional


def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts YouTube video ID from common URL formats:
      https://www.youtube.com/watch?v=ID
      https://youtu.be/ID
      https://www.youtube.com/shorts/ID
    Returns None if pattern does not match.
    """
    try:
        parsed = urllib.parse.urlparse(url.strip())
        host = parsed.netloc.lower().replace("www.", "")
        if host == "youtu.be":
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid if vid else None
        if host in ("youtube.com", "m.youtube.com"):
            if "/shorts/" in parsed.path:
                return parsed.path.split("/shorts/")[1].split("/")[0]
            qs = urllib.parse.parse_qs(parsed.query)
            ids = qs.get("v", [])
            return ids[0] if ids else None
    except Exception:
        return None
    return None


def seconds_to_display(seconds: int) -> str:
    """Converts raw seconds to display string: 147 -> '2:27', 3672 -> '1:01:12'"""
    s = int(seconds)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def build_timestamp_link(video_id: str, seconds: int) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def make_timestamp_result(video_id: str, seconds: int) -> dict:
    s = int(seconds)
    return {
        "seconds": s,
        "display": seconds_to_display(s),
        "link": build_timestamp_link(video_id, s),
    }
