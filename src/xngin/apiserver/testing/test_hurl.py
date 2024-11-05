from xngin.apiserver.testing.hurl import Hurl


def test_hurl():
    assert Hurl.from_script(
        """GET /foo/bar\nH1: v1\nHTTP 200\n```json\ncontents\n```"""
    ) == Hurl(
        method="GET",
        url="/foo/bar",
        headers={"H1": "v1"},
        expected_response="contents",
        expected_status=200,
    )
    assert Hurl.from_script(
        """GET /foo/bar\nHTTP 200\n```json\ncontents\n```"""
    ) == Hurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_response="contents",
        expected_status=200,
    )
    assert Hurl.from_script(
        """GET /foo/bar\n```json\npayload\n```\nHTTP 200\n```json\ncontents\n```"""
    ) == Hurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_response="contents",
        expected_status=200,
        body="payload",
    )
    assert Hurl.from_script("""GET http://localhost:8000/strata?unit_type=test_unit_type
```json
requestbody
```
HTTP 200
```json
[
  { "k": "v" }
```

""") == Hurl(
        method="GET",
        url="http://localhost:8000/strata?unit_type=test_unit_type",
        headers={},
        expected_status=200,
        expected_response='[\n  { "k": "v" }',
        body="requestbody",
    )
    assert Hurl.from_script("""GET http://localhost:8000/strata?unit_type=test_unit_type
Config-ID: testing
```json
requestbody
```
HTTP 200
```json
[
  { "k": "v" }
```

""") == Hurl(
        method="GET",
        url="http://localhost:8000/strata?unit_type=test_unit_type",
        headers={"Config-ID": "testing"},
        expected_status=200,
        expected_response='[\n  { "k": "v" }',
        body="requestbody",
    )
    # Test case for multi-line JSON payload and response
    assert Hurl.from_script("""GET /foo/bar
```json
{
  "payload": "value",
  "multiline": true
}
```
HTTP 200
```json
{
  "response": "data",
  "success": true
}
```
""") == Hurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_status=200,
        expected_response='{\n  "response": "data",\n  "success": true\n}',
        body='{\n  "payload": "value",\n  "multiline": true\n}',
    )
