import re

import pytest

from xngin.apiserver.dwh.field_aliases import FieldAliases


def alias_of(field_name: str) -> str:
    return FieldAliases([field_name]).alias(field_name)


@pytest.mark.parametrize(
    ("field_name", "readable"),
    [
        ("revenue", "m_revenue_"),
        ("Revenue.Total (USD)", "m_revenue_total_usd_"),
        ("participant_id", "m_participant_id_"),
        ("__x__", "m_x_"),
        ("数据", "m_"),
        ("x" * 100, "m_" + "x" * 32 + "_"),
    ],
)
def test_aliases_are_readable_and_need_no_quoting(field_name: str, readable: str):
    alias = alias_of(field_name)
    assert re.fullmatch(re.escape(readable) + "[0-9a-f]{16}", alias)
    assert len(alias + "__stddev") <= 63


def test_names_that_read_alike_have_distinct_aliases():
    names = ["a.b", "a_b", "a b", "A_B", "a__b"]
    assert len(set(FieldAliases(names).aliases())) == len(names)


def test_aliases_are_in_the_order_names_were_first_given():
    aliases = FieldAliases(["rows", "a.b", "rows"])
    assert list(aliases.name_to_aliases()) == [("rows", alias_of("rows")), ("a.b", alias_of("a.b"))]
    assert aliases.aliases() == [alias_of("rows"), alias_of("a.b")]


def test_aliases_map_back_to_their_names():
    aliases = FieldAliases(["rows", "a.b"])
    assert [aliases.name(alias) for alias in aliases.aliases()] == ["rows", "a.b"]


def test_unknown_names_and_aliases_are_refused():
    aliases = FieldAliases(["rows"])
    with pytest.raises(KeyError, match="'cols' is not one of the aliased fields"):
        aliases.alias("cols")
    with pytest.raises(KeyError, match="'rows' is not the alias"):
        aliases.name("rows")
