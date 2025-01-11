from xngin.apiserver import constants
from xngin.apiserver.testing.xurl import Xurl


def test_hurl():
    assert Xurl.from_script(
        """GET /foo/bar\nH1: v1\nHTTP 200\n```json\ncontents\n```"""
    ) == Xurl(
        method="GET",
        url="/foo/bar",
        headers={"H1": "v1"},
        expected_response="contents",
        expected_status=200,
        trailer=None,
    )
    assert Xurl.from_script(
        """GET /foo/bar\nHTTP 200\n```json\ncontents\n```"""
    ) == Xurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_response="contents",
        expected_status=200,
        trailer=None,
    )
    assert Xurl.from_script(
        """GET /foo/bar\n```json\npayload\n```\nHTTP 200\n```json\ncontents\n```"""
    ) == Xurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_response="contents",
        expected_status=200,
        body="payload",
        trailer=None,
    )
    assert Xurl.from_script("""GET http://localhost:8000/strata?participant_type=test_participant_type
```json
requestbody
```
HTTP 200
```json
[
  { "k": "v" }
```

""") == Xurl(
        method="GET",
        url="http://localhost:8000/strata?participant_type=test_participant_type",
        headers={},
        expected_status=200,
        expected_response='[\n  { "k": "v" }',
        body="requestbody",
        trailer=None,
    )
    assert Xurl.from_script("""GET http://localhost:8000/strata?participant_type=test_participant_type
Config-ID: testing
```json
requestbody
```
HTTP 200
```json
[
  { "k": "v" }
```

""") == Xurl(
        method="GET",
        url="http://localhost:8000/strata?participant_type=test_participant_type",
        headers={constants.HEADER_CONFIG_ID: "testing"},
        expected_status=200,
        expected_response='[\n  { "k": "v" }',
        body="requestbody",
        trailer=None,
    )
    # Test case for multi-line JSON payload and response
    assert Xurl.from_script("""GET /foo/bar
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
""") == Xurl(
        method="GET",
        url="/foo/bar",
        headers={},
        expected_status=200,
        expected_response='{\n  "response": "data",\n  "success": true\n}',
        body='{\n  "payload": "value",\n  "multiline": true\n}',
        trailer=None,
    )


def test_hurl_trailer():
    assert Xurl.from_script(
        """GET /foo/bar\nH1: v1\nHTTP 200\n```json\ncontents\n```trailer"""
    ) == Xurl(
        method="GET",
        url="/foo/bar",
        headers={"H1": "v1"},
        expected_response="contents",
        expected_status=200,
        trailer="trailer",
    )

    assert Xurl.from_script(
        """GET /foo/bar\nH1: v1\nHTTP 200\n```json\ncontents\n```trailer1\ntrailer2\n"""
    ) == Xurl(
        method="GET",
        url="/foo/bar",
        headers={"H1": "v1"},
        expected_response="contents",
        expected_status=200,
        trailer="trailer1\ntrailer2\n",
    )
