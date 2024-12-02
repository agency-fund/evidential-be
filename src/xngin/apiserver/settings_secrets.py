import os
import re
from collections.abc import Callable

# Defines a capturing regex for ${src:ref} pairs.
SECRET_REF = re.compile(
    re.escape("${") + r"(?P<src>secret):(?P<ref>[a-zA-Z0-9_]+)" + re.escape("}")
)


class MissingSecretException(Exception):
    def __init__(self, variable):
        super().__init__(f'Secret "{variable}" could not be found.')


def replace_secrets(v):
    """Replaces references to secrets in v with their actual value."""
    # Find all the references that might exist
    variables = _search_for_variables(v)
    # Resolve the secrets. This may involve a remote API call.
    resolved = _resolve_secrets(variables)
    # Replace the references.
    return _replace_variables(v, resolved)


def _search_for_variables(tmpl) -> set[str]:
    """Returns a set containing unique replaceable values found in tmpl."""

    def visitor_onepassword_refs(s):
        if not isinstance(s, str):
            return set()
        return {(m.group("src"), m.group("ref")) for m in SECRET_REF.finditer(s)}

    return walk_unique(tmpl, visitor_onepassword_refs)


def walk_unique(node, visitor: Callable[[str], set[str]]) -> set[str]:
    """Invokes visitor on every node and returns the unique return values from the visitor."""

    if isinstance(node, str):
        return visitor(node)
    if isinstance(node, list | tuple):
        ss = [walk_unique(i, visitor) for i in node]
        return {i for v in ss for i in v}
    if isinstance(node, dict):
        ss = [walk_unique(i, visitor) for i in node.values()]
        return {i for v in ss for i in v}
    return set()


def _resolve_secrets(variables):
    # We only want to resolve secrets.
    variables = [v for v in variables if v[0] == "secret"]
    # We only support environment variables for secrets; in the future, this might be AWS Secrets Manager, files,
    # BitWarden, etc.
    if os.environ.get("XNGIN_SECRETS_SOURCE", "environ") != "environ":
        raise ValueError("XNGIN_SECRETS_SOURCE is misconfigured.")
    replacements = {}
    for source, name in variables:
        try:
            replacements[source, name] = os.environ[name]
        except KeyError as exc:
            raise MissingSecretException(name) from exc
    return replacements


def _replace_variables(tmpl, variables):
    """Replaces all ${SRC:REF} references with the corresponding value from the variables dict."""

    def visitor_var_subst(s):
        if not isinstance(s, str):
            return s
        return re.sub(
            SECRET_REF,
            lambda match: variables[match.group("src"), match.group("ref")],
            s,
        )

    return _produce_strings(tmpl, visitor_var_subst)


def _produce_strings(node, visitor):
    """Builds a new Python value from t by replacing any string values with the return value of the visitor."""
    if isinstance(node, str):
        return visitor(node)
    if isinstance(node, list | tuple):
        return [_produce_strings(item, visitor) for item in node]
    if isinstance(node, dict):
        return {k: _produce_strings(v, visitor) for k, v in node.items()}
    if isinstance(node, bool | int | float | type(None)):
        return node
    raise ValueError(f"Unsupported type: {type(node)}")
