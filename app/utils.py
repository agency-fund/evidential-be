STRIP_UNSAFE_HEADER_CHARS = str.maketrans("", "", "\r\n")


def safe_for_headers(v: str) -> str:
    return v.translate(STRIP_UNSAFE_HEADER_CHARS)
