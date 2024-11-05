import re

from typing import Self

from parsy import regex, string, seq, alt, generate, whitespace
from pydantic import BaseModel


class Hurl(BaseModel):
    """Simple parser for a Hurl-like request/response format.

    The format of the input is line-oriented::

        GET /foo/bar   # required
        Header: Value  # may be repeated or omitted
        ```json        # may be omitted
        request body
        ```
        HTTP 200       # required
        ```json        # required
        expected response
        ```

    """

    method: str
    url: str
    headers: dict[str, str]
    expected_status: int
    expected_response: str | None = None
    body: str | None = None

    def to_script(self):
        """Generates a Hurl script from this Hurl instance."""
        return "\n".join([
            line
            for line in [
                f"{self.method} {self.url}",
                *[f"{k}: {v}" for k, v in sorted(self.headers.items())],
                f"```json\n{self.body}\n```" if self.body else None,
                f"HTTP {self.expected_status}",
                f"```json\n{self.expected_response}\n```",
            ]
            if line is not None
        ])

    @staticmethod
    def from_script(script: str) -> Self:
        """Constructs a Hurl from a string."""

        @generate
        def parser():
            method = yield string("GET") | string("POST")
            yield string(" ")
            path = yield regex(r"[^\n]+") << string("\n")
            headers = (
                yield regex(r"([^:\n]+): ([^:\n]*)\n", group=(1, 2))
                .many()
                .map(dict)
                .optional()
            )
            json_block = regex(r"```json\n(.+?)\n```\n+", flags=re.DOTALL, group=1)
            status_code_p = regex(r"HTTP (\d+)\n+", group=1)
            json_block_eof = regex(r"```json\n(.+)\n```", flags=re.DOTALL, group=1)
            payloads = yield (
                alt(
                    seq(
                        req_body=json_block,
                        status=status_code_p,
                        resp_body=json_block_eof,
                    ),
                    seq(status=status_code_p, resp_body=json_block_eof),
                    seq(resp_body=json_block_eof),
                )
                << whitespace.optional()
            )

            return Hurl(  # noqa: B901
                method=method,
                url=path,
                headers=headers,
                expected_response=payloads.get("resp_body"),
                body=payloads.get("req_body"),
                expected_status=payloads.get("status"),
            )

        return parser.parse(script)
