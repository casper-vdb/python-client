from __future__ import annotations

import struct
from typing import Any, Iterable, Sequence

from casper_client.errors import CasperError, CasperErrorKind
from casper_client.http import ApiClient
from casper_client.models import (
    AppConfig,
    ClusterInfo,
    CollectionInfo,
    CollectionsListResponse,
    GetVectorResponse,
    IndexPolicyPatch,
    IndexStorageSpec,
    SearchResult,
    VectorPoint,
)

# One (u64 id, f32 score) record of a binary search / scores body.
_SCORED_RECORD_SIZE = 12

# How long remove_peer waits: longer than the server's own 30 s wait for the
# removal to commit, so that its 504 reaches the caller rather than a client
# timeout.
_REMOVE_PEER_TIMEOUT = 60.0


def _coerce_points(points: Iterable[VectorPoint | dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in points:
        if isinstance(p, VectorPoint):
            out.append(p.to_dict())
        elif isinstance(p, dict):
            out.append({"id": int(p["id"]), "vector": list(p["vector"])})
        else:
            raise CasperError(
                kind=CasperErrorKind.CLIENT,
                message=f"invalid point: expected VectorPoint or dict, got {type(p).__name__}",
            )
    return out


class CasperClient:
    """High-level HTTP client for Casper Vector Database."""

    def __init__(
        self,
        *,
        host: str = "http://127.0.0.1",
        http_port: int = 7222,
        timeout: float | None = 30.0,
        api_key: str | None = None,
        **httpx_kwargs: Any,
    ) -> None:
        base = host.rstrip("/")
        if not (base.startswith("http://") or base.startswith("https://")):
            base = "http://" + base

        self._rest_uri = f"{base}:{http_port}"
        if api_key is not None:
            # Sent on every request; the server ignores it on /health and /readyz.
            headers = dict(httpx_kwargs.pop("headers", None) or {})
            headers["api-key"] = api_key
            httpx_kwargs["headers"] = headers
        self._timeout = timeout
        self._http = ApiClient(self._rest_uri, timeout=timeout, **httpx_kwargs)

    @property
    def rest_uri(self) -> str:
        return self._rest_uri

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CasperClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -------------------------
    # Health / cluster / config / metrics
    # -------------------------

    def health(self) -> None:
        """GET /health — raises on non-2xx (suitable for k8s probes)."""
        self._http.request_bytes(method="GET", url="/health")

    def ready(self, *, quorum: bool = False) -> bool:
        """GET /readyz — whether this node should get reads: False on 503
        (joining, leaderless or stopping). ``quorum=True`` also requires the
        node to count for quorum, for rolling restarts."""
        params = {"quorum": "true"} if quorum else None
        try:
            self._http.request_bytes(method="GET", url="/readyz", params=params)
        except CasperError as e:
            if e.kind == CasperErrorKind.UNAVAILABLE:
                return False
            raise
        return True

    def get_cluster(self) -> ClusterInfo:
        """GET /cluster — node role (standalone/cluster) and, in a cluster,
        the Raft state and membership. Inspect :attr:`ClusterInfo.role` to
        know which fields are meaningful."""
        data = self._http.request_json(method="GET", url="/cluster")
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /cluster",
            )
        return ClusterInfo.from_dict(data)

    def remove_peer(self, node_id: str) -> str:
        """DELETE /cluster/peer/{node_id} — remove a node from the cluster
        membership; must reach the leader. Returns the removed node id."""
        timeout = None if self._timeout is None else max(self._timeout, _REMOVE_PEER_TIMEOUT)
        data = self._http.request_json(
            method="DELETE",
            url="/cluster/peer/{node_id}",
            path_params={"node_id": node_id},
            timeout=timeout,
        )
        if not isinstance(data, dict) or not isinstance(data.get("removed"), str):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected {'removed': ...} from /cluster/peer/{node_id}",
            )
        return data["removed"]

    def set_transfer_limit(self, mb_per_sec: int) -> None:
        """PUT /cluster/transfer-limit — rate limit of the index and WAL
        transfers this node sends. Held in memory only: a restart returns to
        the configured value."""
        self._http.request_bytes(
            method="PUT",
            url="/cluster/transfer-limit",
            json={"mb_per_sec": int(mb_per_sec)},
            headers={"Content-Type": "application/json"},
        )

    def get_config(self) -> AppConfig:
        """GET /config — runtime ``AppConfig`` the server started with.
        Immutable after startup; restart the server to change a value."""
        data = self._http.request_json(method="GET", url="/config")
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /config",
            )
        return AppConfig.from_dict(data)

    def get_metrics(self) -> str:
        """GET /metrics — Prometheus metrics in the text exposition format."""
        raw = self._http.request_bytes(method="GET", url="/metrics")
        return raw.decode("utf-8", errors="replace")

    # -------------------------
    # Collections
    # -------------------------

    def list_collections(self) -> CollectionsListResponse:
        """GET /collections — names only (with auth, the ones the credential
        can see); use get_collection for details."""
        data = self._http.request_json(method="GET", url="/collections")
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /collections",
            )
        return CollectionsListResponse.from_dict(data)

    def get_collection(self, name: str) -> CollectionInfo:
        """GET /collections/{name} — metadata, WAL progress and
        ``index.state``."""
        data = self._http.request_json(
            method="GET", url="/collections/{name}", path_params={"name": name}
        )
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /collections/{name}",
            )
        return CollectionInfo.from_dict(data)

    def create_collection(
        self,
        name: str,
        *,
        dim: int,
        max_size: int,
        metric: str,
        quantization: str,
        m: int,
        m0: int,
        ef_construction: int,
        normalization: bool | None = None,
        storage: IndexStorageSpec | None = None,
    ) -> str:
        """POST /collection/{name} — create the collection with the schema and
        policy of its index; returns the collection id. No index is built:
        call start_indexing. The schema never changes; the policy can, with
        update_index_policy."""
        index: dict[str, Any] = {
            "hnsw": {
                "metric": metric,
                "quantization": quantization,
                "m": int(m),
                "m0": int(m0),
                "ef_construction": int(ef_construction),
            },
        }
        if normalization is not None:
            index["normalization"] = bool(normalization)
        if storage is not None:
            index["storage"] = storage.to_dict()

        body = {"dim": int(dim), "max_size": int(max_size), "index": index}
        data = self._http.request_json(
            method="POST",
            url="/collection/{name}",
            path_params={"name": name},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        if not isinstance(data, dict) or not isinstance(data.get("id"), str):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected {'id': ...} from /collection/{name}",
            )
        return data["id"]

    def delete_collection(self, name: str) -> None:
        self._http.request_bytes(
            method="DELETE", url="/collection/{name}", path_params={"name": name}
        )

    # -------------------------
    # Vectors
    # -------------------------

    def get_vector(self, collection: str, *, id: int) -> GetVectorResponse | None:
        """GET /collection/{name}/vector/{id} — the vector as the index holds
        it (see :class:`GetVectorResponse` for the i8 form), or None if the
        index does not hold ``id``. A missing collection raises NOT_FOUND."""
        try:
            data = self._http.request_json(
                method="GET",
                url="/collection/{name}/vector/{id}",
                path_params={"name": collection, "id": int(id)},
            )
        except CasperError as e:
            # Only the JSON 404 is "no such vector"; a missing collection is plain text.
            vector_missing = (
                e.kind == CasperErrorKind.NOT_FOUND
                and e.details is not None
                and e.details.get("error") == "Vector not found"
            )
            if vector_missing:
                return None
            raise
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /collection/{name}/vector/{id}",
            )
        return GetVectorResponse.from_dict(data)

    def batch_update(
        self,
        collection: str,
        *,
        upserts: Iterable[VectorPoint | dict[str, Any]] | None = None,
        delete: Sequence[int] | None = None,
    ) -> None:
        """POST /collection/{name}/update — the one write: upserts by id, then
        deletes, as one WAL record. Either may be empty, not both."""
        payload = {
            "upserts": _coerce_points(upserts) if upserts is not None else [],
            "delete": [int(i) for i in delete] if delete is not None else [],
        }
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/update",
            path_params={"name": collection},
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    # -------------------------
    # Indexes
    # -------------------------

    def start_indexing(self, collection: str) -> None:
        """POST /collection/{name}/index/start — start building the index on
        the schema the collection was created with (202; the build runs in
        the background). Poll get_collection().index.state: READY when done,
        NOT_STARTED again after a failed build (standalone). A repeated call
        changes nothing."""
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/index/start",
            path_params={"name": collection},
        )

    def update_index_policy(self, collection: str, policy: IndexPolicyPatch) -> None:
        """PATCH /collection/{name}/index — change the snapshot / compaction
        policy of the index, before indexing starts as well; only the keys
        the patch names change."""
        self._http.request_bytes(
            method="PATCH",
            url="/collection/{name}/index",
            path_params={"name": collection},
            json={"storage": policy.to_dict()},
            headers={"Content-Type": "application/json"},
        )

    def trigger_snapshot(self, collection: str) -> None:
        """POST /collection/{name}/index/snapshot/trigger — snapshot the
        index on this node now, regardless of thresholds (202)."""
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/index/snapshot/trigger",
            path_params={"name": collection},
        )

    def trigger_compaction(self, collection: str) -> None:
        """POST /collection/{name}/index/compaction/trigger — start a fold
        pass on this node now; it folds only the ranges the policy's rules
        hold for (202)."""
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/index/compaction/trigger",
            path_params={"name": collection},
        )

    # -------------------------
    # Search
    # -------------------------

    def search(
        self,
        collection: str,
        *,
        query: Sequence[float],
        limit: int,
        ef_search: int | None = None,
        output: str = "bin",
    ) -> list[SearchResult]:
        if output not in ("bin", "json"):
            raise CasperError(
                kind=CasperErrorKind.CLIENT, message="output must be 'bin' or 'json'"
            )

        params: dict[str, Any] = {"limit": int(limit), "output": output}
        if ef_search is not None:
            params["ef_search"] = int(ef_search)

        return self._post_scored(
            "/collection/{name}/search", collection, params, {"vector": list(query)}
        )

    def scores(
        self,
        collection: str,
        *,
        query: Sequence[float],
        ids: Sequence[int],
        output: str = "bin",
    ) -> list[SearchResult]:
        """POST /collection/{name}/scores — the score of ``query`` against
        each of ``ids``, as a search would score it, in the order of ``ids``.
        Ids the index does not hold are left out; an id given twice is
        answered twice."""
        if output not in ("bin", "json"):
            raise CasperError(
                kind=CasperErrorKind.CLIENT, message="output must be 'bin' or 'json'"
            )

        body = {"vector": list(query), "ids": [int(i) for i in ids]}
        return self._post_scored("/collection/{name}/scores", collection, {"output": output}, body)

    def _post_scored(
        self, url: str, collection: str, params: dict[str, Any], body: dict[str, Any]
    ) -> list[SearchResult]:
        if params["output"] == "bin":
            raw = self._http.request_bytes(
                method="POST",
                url=url,
                path_params={"name": collection},
                params=params,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            return self._decode_search_bin(raw)

        data = self._http.request_json(
            method="POST",
            url=url,
            path_params={"name": collection},
            params=params,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        items = data.get("result") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected {'result': [...]} from search JSON response",
            )
        out: list[SearchResult] = []
        for item in items:
            if isinstance(item, dict) and "id" in item and "score" in item:
                out.append(SearchResult(id=int(item["id"]), score=float(item["score"])))
        return out

    @staticmethod
    def _decode_search_bin(buf: bytes) -> list[SearchResult]:
        """A u32 LE count, then that many 12-byte records: u64 id, f32 score."""
        if len(buf) < 4:
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="binary search response too short",
            )
        (count,) = struct.unpack_from("<I", buf, 0)
        expected = 4 + int(count) * _SCORED_RECORD_SIZE
        if len(buf) < expected:
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message=(
                    f"binary search response truncated: expected at least "
                    f"{expected} bytes, got {len(buf)}"
                ),
            )
        out: list[SearchResult] = []
        offset = 4
        for _ in range(int(count)):
            id_u64, score_f32 = struct.unpack_from("<Qf", buf, offset)
            out.append(SearchResult(id=int(id_u64), score=float(score_f32)))
            offset += _SCORED_RECORD_SIZE
        return out
