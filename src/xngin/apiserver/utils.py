import copy

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
