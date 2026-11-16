from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from casper_client.errors import CasperError, CasperErrorKind, classify_http_error


def _parse_retry_after(value: str | None) -> int | None:
    """`Retry-After` in seconds; Casper never sends the HTTP-date form."""
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


class ApiClient:
    """
    Minimal httpx-based transport inspired by qdrant_client.http.ApiClient.
    """

    def __init__(self, host: str, *, timeout: float | None = 30.0, **kwargs: Any) -> None:
        self.host = host
        # Avoid failing on exotic proxy schemes in env (e.g. socks://) unless user opts in.
        if "trust_env" not in kwargs:
            kwargs["trust_env"] = False
        self._client = httpx.Client(timeout=timeout, **kwargs)

    def close(self) -> None:
        self._client.close()

    def _build_url(self, url: str, path_params: dict[str, Any] | None = None) -> str:
        if path_params is None:
            path_params = {}
        host = self.host if self.host.endswith("/") else self.host + "/"
        url = url[1:] if url.startswith("/") else url
        # Each parameter is one path segment, so `/`, `?` and `#` in a name are escaped.
        escaped = {key: quote(str(value), safe="") for key, value in path_params.items()}
        return urljoin(host, url.format(**escaped))

    def request_bytes(
        self,
        *,
        method: str,
        url: str,
        path_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bytes:
        full_url = self._build_url(url, path_params)
        req = self._client.build_request(method, full_url, **kwargs)
        try:
            resp = self._client.send(req)
        except httpx.TransportError as e:
            raise CasperError(kind=CasperErrorKind.TRANSPORT, message="request failed", cause=e) from e
        except Exception as e:
            raise CasperError(kind=CasperErrorKind.CLIENT, message="request failed", cause=e) from e

        content = resp.content
        if 200 <= resp.status_code < 300:
            return content

        msg = ""
        parsed: Any = None
        if content:
            # Casper often returns {"error": "..."}; fall back to raw bytes.
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("error"), str) and parsed.get("error"):
                msg = parsed["error"]
            else:
                msg = content.decode("utf-8", errors="replace")

        if not msg:
            msg = resp.reason_phrase or "request failed"

        # The one 405 with a JSON body is "not the Raft leader"; the others
        # (already exists, build in progress) are plain text.
        # `leader_addr` is absent while no leader is known.
        not_leader = resp.status_code == 405 and isinstance(parsed, dict) and "error" in parsed
        leader_addr = parsed.get("leader_addr") if not_leader else None

        raise classify_http_error(
            resp.status_code,
            msg,
            retry_after=_parse_retry_after(resp.headers.get("Retry-After")),
            not_leader=not_leader,
            leader_addr=leader_addr if isinstance(leader_addr, str) else None,
            details=parsed if isinstance(parsed, dict) else None,
        )

    def request_json(
        self,
        *,
        method: str,
        url: str,
        path_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raw = self.request_bytes(method=method, url=url, path_params=path_params, **kwargs)
        if raw == b"":
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message=f"failed to parse JSON response: {e}",
                cause=e,
            ) from e


