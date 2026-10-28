"""Aliases for customer field names in the queries and DataFrames we build.

A field name may be any string a warehouse allows, including one we use ourselves, such as participant_id, or one a
backend would rewrite or truncate. Queries and DataFrames that hold a field's values alongside other columns label
them by alias instead.
"""

import string
from collections.abc import Iterable, Iterator

import xxhash

# An alias's readable part is at most this long; with the 16-character digest, an alias plus a suffix such as
# "__stddev" stays under Postgres's 63-byte limit on identifiers.
READABLE_LENGTH = 32

# Maps each byte of an alias's readable part to itself if it is a lowercase letter or digit, else to a space. Every
# byte of a UTF-8 encoded non-ASCII character becomes a space.
_READABLE_BYTES = bytes(b if chr(b) in string.ascii_lowercase + string.digits else ord(" ") for b in range(256))


def _alias(field_name: str) -> str:
    # Joins the runs of lowercase letters and digits with underscores; much faster than the equivalent regex.
    readable = b"_".join(field_name.lower().encode().translate(_READABLE_BYTES).split())[:READABLE_LENGTH].decode()
    digest = xxhash.xxh3_64_hexdigest(field_name.encode())
    return f"m_{readable}_{digest}" if readable else f"m_{digest}"


class FieldAliases:
    """The aliases of a set of field names, in the order the names were first given.

    An alias is readable, needs no quoting on any backend, and no other field name has it: it is lowercase letters,
    digits and underscores, starting with "m_". "Revenue.Total (USD)" becomes "m_revenue_total_usd_" followed by a
    16-character digest of the full name.
    """

    def __init__(self, field_names: Iterable[str]):
        """Raises ValueError if two of field_names have the same alias."""
        self._name_to_alias = {field_name: _alias(field_name) for field_name in field_names}
        self._alias_to_name = {alias: field_name for field_name, alias in self._name_to_alias.items()}
        if len(self._alias_to_name) != len(self._name_to_alias):
            raise ValueError(f"Field names {sorted(self._name_to_alias)} do not have distinct aliases.")

    def alias(self, field_name: str) -> str:
        """Raises KeyError if field_name is not one of these fields."""
        if (alias := self._name_to_alias.get(field_name)) is None:
            raise KeyError(f"{field_name!r} is not one of the aliased fields {list(self._name_to_alias)}.")
        return alias

    def name(self, alias: str) -> str:
        """Raises KeyError if alias is not the alias of one of these fields."""
        if (field_name := self._alias_to_name.get(alias)) is None:
            raise KeyError(f"{alias!r} is not the alias of any of the fields {list(self._name_to_alias)}.")
        return field_name

    def aliases(self) -> list[str]:
        """The aliases, in the order the names were first given."""
        return list(self._name_to_alias.values())

    def name_to_aliases(self) -> Iterator[tuple[str, str]]:
        """Each field name and its alias, in the order the names were first given."""
        return iter(self._name_to_alias.items())
