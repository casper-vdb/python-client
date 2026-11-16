from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CasperErrorKind(str, Enum):
    CLIENT = "client"
    SERVER = "server"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    OPERATION_NOT_ALLOWED = "operation_not_allowed"
    # A mutation reached a cluster node that is not the Raft leader;
    # CasperError.leader_addr says where to retry, when a leader is known.
    NOT_LEADER = "not_leader"
    CONFLICT = "conflict"
    # The batch is over the server's limit; split it.
    PAYLOAD_TOO_LARGE = "payload_too_large"
    # The write queue is full; see CasperError.retry_after.
    TOO_MANY_REQUESTS = "too_many_requests"
    # The node cannot serve the request yet; see CasperError.retry_after.
    UNAVAILABLE = "unavailable"
    # The write's outcome is unknown; retrying is safe (writes are upserts by id).
    TIMEOUT = "timeout"
    INSUFFICIENT_STORAGE = "insufficient_storage"
    # No HTTP answer came back: the connection failed or the request timed
    # out; CasperError.cause holds the httpx error. A write may still have
    # been applied.
    TRANSPORT = "transport"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


@dataclass
class CasperError(Exception):
    kind: CasperErrorKind
    message: str
    status: int | None = None
    cause: Exception | None = None
    # Seconds from the `Retry-After` header (sent with 429 and 503), if any.
    retry_after: int | None = None
    # The Raft leader's address on a NOT_LEADER error; None while no leader is known.
    leader_addr: str | None = None
    # The error body when the server answered with a JSON object; None for plain text.
    details: dict | None = None

    def __str__(self) -> str:
        status_part = f" (status={self.status})" if self.status is not None else ""
        leader_part = f" (leader_addr={self.leader_addr})" if self.leader_addr is not None else ""
        retry_part = f" (retry_after={self.retry_after}s)" if self.retry_after is not None else ""
        cause_part = f" (cause={self.cause!r})" if self.cause is not None else ""
        return f"casper: {self.kind.value}{status_part}: {self.message}{leader_part}{retry_part}{cause_part}"


_KIND_BY_STATUS = {
    401: CasperErrorKind.UNAUTHORIZED,
    403: CasperErrorKind.FORBIDDEN,
    404: CasperErrorKind.NOT_FOUND,
    405: CasperErrorKind.OPERATION_NOT_ALLOWED,
    409: CasperErrorKind.CONFLICT,
    413: CasperErrorKind.PAYLOAD_TOO_LARGE,
    429: CasperErrorKind.TOO_MANY_REQUESTS,
    503: CasperErrorKind.UNAVAILABLE,
    504: CasperErrorKind.TIMEOUT,
    507: CasperErrorKind.INSUFFICIENT_STORAGE,
}


def classify_http_error(
    status: int,
    message: str,
    *,
    retry_after: int | None = None,
    not_leader: bool = False,
    leader_addr: str | None = None,
    details: dict | None = None,
) -> CasperError:
    if not_leader:
        kind = CasperErrorKind.NOT_LEADER
    elif status in _KIND_BY_STATUS:
        kind = _KIND_BY_STATUS[status]
    elif 400 <= status < 500:
        kind = CasperErrorKind.CLIENT
    elif 500 <= status <= 599:
        kind = CasperErrorKind.SERVER
    else:
        kind = CasperErrorKind.UNKNOWN
    return CasperError(
        kind=kind,
        status=status,
        message=message,
        retry_after=retry_after,
        leader_addr=leader_addr,
        details=details,
    )
