from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HnswIndexInfo:
    metric: str
    quantization: str
    m: int
    m0: int
    ef_construction: int

    @staticmethod
    def from_dict(data: dict) -> "HnswIndexInfo":
        return HnswIndexInfo(
            metric=data.get("metric", ""),
            quantization=data.get("quantization", ""),
            m=int(data.get("m", 0)),
            m0=int(data.get("m0", 0)),
            ef_construction=int(data.get("ef_construction", 0)),
        )


@dataclass
class IndexInfo:
    normalization: bool
    hnsw: HnswIndexInfo | None = None

    @staticmethod
    def from_dict(data: dict) -> "IndexInfo":
        hnsw = data.get("hnsw")
        return IndexInfo(
            normalization=bool(data.get("normalization", False)),
            hnsw=HnswIndexInfo.from_dict(hnsw) if isinstance(hnsw, dict) else None,
        )


@dataclass
class CollectionInfo:
    name: str
    dimension: int
    mutable: bool
    has_index: bool
    max_size: int
    size: int
    status: str
    index: IndexInfo | None = None

    @staticmethod
    def from_dict(data: dict) -> "CollectionInfo":
        idx = data.get("index")
        return CollectionInfo(
            name=data.get("name", ""),
            dimension=int(data.get("dimension", 0)),
            mutable=bool(data.get("mutable", False)),
            has_index=bool(data.get("has_index", False)),
            max_size=int(data.get("max_size", 0)),
            size=int(data.get("size", 0)),
            status=str(data.get("status", "")),
            index=IndexInfo.from_dict(idx) if isinstance(idx, dict) else None,
        )


@dataclass
class CollectionsListResponse:
    collections: list[CollectionInfo] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict) -> "CollectionsListResponse":
        cols = data.get("collections", [])
        if not isinstance(cols, list):
            cols = []
        return CollectionsListResponse(
            collections=[CollectionInfo.from_dict(c) for c in cols if isinstance(c, dict)]
        )


@dataclass
class SearchResult:
    id: int
    score: float


@dataclass
class VectorPoint:
    """A single (id, vector) pair used in upsert/batch operations."""

    id: int
    vector: list[float]

    def to_dict(self) -> dict:
        return {"id": int(self.id), "vector": list(self.vector)}


@dataclass
class GetVectorResponse:
    id: int
    vector: list[float]

    @staticmethod
    def from_dict(data: dict) -> "GetVectorResponse":
        vec = data.get("vector")
        return GetVectorResponse(
            id=int(data.get("id", 0)),
            vector=list(vec) if isinstance(vec, list) else [],
        )
