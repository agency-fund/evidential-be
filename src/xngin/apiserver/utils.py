import copy
from urllib.parse import urlparse, urlunparse, quote

STRIP_UNSAFE_HEADER_CHARS = str.maketrans("", "", "\r\n")


def safe_for_headers(v: str) -> str:
    return v.translate(STRIP_UNSAFE_HEADER_CHARS)


def merge_dicts(left: dict, right: dict):
    """Merges the left and right dictionaries into a new dictionary.

    If the value of a key in both left and right are a dict, they are recursively merged.
    Otherwise, the value from right will overwrite the value for that key.
    """
    result = copy.deepcopy(left)
    for key, rvalue in right.items():
        if key in result and isinstance(left[key], dict) and isinstance(rvalue, dict):
            result[key] = merge_dicts(left[key], rvalue)
        else:
            result[key] = rvalue
    return result


def substitute_url(url_template: str, raw_replacements: dict[str, str]):
    """Sub placeholders using curly brackets {key} with un-escaped replacement values.

    Substitutions are done using python's string format() after escaping the values.
    """

    parsed = urlparse(url_template)
    # Don't want users putting '/' in any values, hence safe=''
    safe_replacements = {k: quote(v, safe="") for k, v in raw_replacements.items()}
    # Replace any placeholders in the path and query parameters
    new_path = parsed.path.format(**safe_replacements)
    # TODO: if ultra-picky, could parse_qs() the .query and look to replace
    # values with urlencode()'ed replacements instead of the quote() above.
    new_query = parsed.query.format(**safe_replacements)

    # Return the reconstructed url
    return urlunparse(parsed._replace(path=new_path, query=new_query))
