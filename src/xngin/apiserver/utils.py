import copy
import random
import secrets
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
    """
    Replace placeholder values in url_template with values from raw_replacements.

    Placeholders are replaced with properly escaped values.
    Placeholders may in the path or query string.

    The returned URL is guaranteed to be safe; i.e. all values in
    raw_replacements are quoted appropriately in paths and query parameters.
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


def random_choice(choices: list, seed: int | None = None):
    """Choose a random value from choices."""
    if seed:
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        # use a predictable random
        r = random.Random(seed)
        return r.choice(choices)
    # Use very strong random by default
    return secrets.choice(choices)
