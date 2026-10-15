"""Accept spreadsheet links, never arbitrary URLs for the connector to fetch."""

import re
from urllib.parse import parse_qs, urlsplit


def parse_spreadsheet_url(value: str) -> tuple[str, int | None]:
    parsed = urlsplit(value)
    match = re.fullmatch(r"/spreadsheets/d/([a-zA-Z0-9_-]+)(?:/.*)?", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "docs.google.com" or match is None:
        raise ValueError("Use a Google Sheets URL starting with https://docs.google.com/spreadsheets/d/.")
    params = parse_qs(parsed.fragment, keep_blank_values=True) | parse_qs(parsed.query, keep_blank_values=True)
    gid = params.get("gid", [None])[0]
    if gid is not None and not re.fullmatch(r"[0-9]+", gid):
        raise ValueError("The spreadsheet tab ID (gid) must be a non-negative integer.")
    return match[1], int(gid) if gid is not None else None
