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
    SearchResult,
    VectorPoint,
)


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
        **httpx_kwargs: Any,
    ) -> None:
        base = host.rstrip("/")
        if not (base.startswith("http://") or base.startswith("https://")):
            base = "http://" + base

        self._rest_uri = f"{base}:{http_port}"
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
    # Health / cluster / config
    # -------------------------

    def health(self) -> None:
        """GET /health — raises on non-2xx (suitable for k8s probes)."""
        self._http.request_bytes(method="GET", url="/health")

    def get_cluster(self) -> ClusterInfo:
        """GET /cluster — node role (standalone/master/replica) and
        replication progress. Inspect :attr:`ClusterInfo.role` to know which
        fields are meaningful."""
        data = self._http.request_json(method="GET", url="/cluster")
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /cluster",
            )
        return ClusterInfo.from_dict(data)

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

    # -------------------------
    # Collections
    # -------------------------

    def list_collections(self) -> CollectionsListResponse:
        data = self._http.request_json(method="GET", url="/collections")
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /collections",
            )
        return CollectionsListResponse.from_dict(data)

    def get_collection(self, name: str) -> CollectionInfo:
        data = self._http.request_json(
            method="GET", url="/collection/{name}", path_params={"name": name}
        )
        if not isinstance(data, dict):
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="expected JSON object from /collection/{name}",
            )
        return CollectionInfo.from_dict(data)

    def create_collection(self, name: str, *, dim: int, max_size: int) -> None:
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}",
            path_params={"name": name},
            params={"dim": dim, "max_size": max_size},
        )

    def delete_collection(self, name: str) -> None:
        self._http.request_bytes(
            method="DELETE", url="/collection/{name}", path_params={"name": name}
        )

    def mute_collection(self, name: str) -> None:
        self._http.request_bytes(
            method="POST", url="/collection/{name}/mute", path_params={"name": name}
        )

    def unmute_collection(self, name: str) -> None:
        self._http.request_bytes(
            method="POST", url="/collection/{name}/unmute", path_params={"name": name}
        )

    # -------------------------
    # Vectors
    # -------------------------

    def upsert_vectors(
        self,
        collection: str,
        vectors: Iterable[VectorPoint | dict[str, Any]],
    ) -> None:
        """PUT /collection/{name}/vectors — upsert a batch of points."""
        payload = {"vectors": _coerce_points(vectors)}
        self._http.request_bytes(
            method="PUT",
            url="/collection/{name}/vectors",
            path_params={"name": collection},
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    def delete_vectors(self, collection: str, ids: Sequence[int]) -> None:
        """POST /collection/{name}/vectors/delete — delete a batch of ids."""
        payload = {"vectors": [int(i) for i in ids]}
        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/vectors/delete",
            path_params={"name": collection},
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    def get_vector(self, collection: str, *, id: int) -> GetVectorResponse:
        data = self._http.request_json(
            method="GET",
            url="/collection/{name}/vector/{id}",
            path_params={"name": collection, "id": id},
        )
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
        insert: Iterable[VectorPoint | dict[str, Any]] | None = None,
        delete: Sequence[int] | None = None,
    ) -> None:
        """POST /collection/{name}/update — combined insert + delete batch."""
        payload = {
            "insert": _coerce_points(insert) if insert is not None else [],
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

    def create_hnsw_index(
        self,
        collection: str,
        *,
        metric: str,
        quantization: str,
        m: int,
        m0: int,
        ef_construction: int,
        normalization: bool | None = None,
    ) -> None:
        """POST /collection/{name}/index — server returns 202; the index
        build runs asynchronously. Poll get_collection().has_index to wait
        for completion."""
        body: dict[str, Any] = {
            "hnsw": {
                "metric": metric,
                "quantization": quantization,
                "m": int(m),
                "m0": int(m0),
                "ef_construction": int(ef_construction),
            },
        }
        if normalization is not None:
            body["normalization"] = bool(normalization)

        self._http.request_bytes(
            method="POST",
            url="/collection/{name}/index",
            path_params={"name": collection},
            json=body,
            headers={"Content-Type": "application/json"},
        )

    def delete_index(self, collection: str) -> None:
        self._http.request_bytes(
            method="DELETE",
            url="/collection/{name}/index",
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

        if output == "bin":
            raw = self._http.request_bytes(
                method="POST",
                url="/collection/{name}/search",
                path_params={"name": collection},
                params=params,
                json={"vector": list(query)},
                headers={"Content-Type": "application/json"},
            )
            return self._decode_search_bin(raw)

        data = self._http.request_json(
            method="POST",
            url="/collection/{name}/search",
            path_params={"name": collection},
            params=params,
            json={"vector": list(query)},
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
        if len(buf) < 4:
            raise CasperError(
                kind=CasperErrorKind.INVALID_RESPONSE,
                message="binary search response too short",
            )
        (count,) = struct.unpack_from("<I", buf, 0)
        expected = 4 + int(count) * 8
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
            id_u32, score_f32 = struct.unpack_from("<If", buf, offset)
            out.append(SearchResult(id=int(id_u32), score=float(score_f32)))
            offset += 8
        return out
