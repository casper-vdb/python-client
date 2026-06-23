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
    # --- Progress / replication cursors (Casper v0.1.0+) ---------------------
    # WAL head: highest durably-fsynced data WAL seq + 1.
    current_seq: int = 0
    # Highest WAL seq applied to the in-memory HNSW. `current_seq -
    # applied_seq` is the applier backlog.
    applied_seq: int = 0
    # WAL seq covered by the on-disk snapshot, if any. None until the first
    # snapshot is taken.
    index_seq: int | None = None
    # Per-collection catalog head (catalog WAL seq + 1).
    catalog_seq: int = 0

    @staticmethod
    def from_dict(data: dict) -> "CollectionInfo":
        idx = data.get("index")
        idx_seq = data.get("index_seq")
        return CollectionInfo(
            name=data.get("name", ""),
            dimension=int(data.get("dimension", 0)),
            mutable=bool(data.get("mutable", False)),
            has_index=bool(data.get("has_index", False)),
            max_size=int(data.get("max_size", 0)),
            size=int(data.get("size", 0)),
            status=str(data.get("status", "")),
            index=IndexInfo.from_dict(idx) if isinstance(idx, dict) else None,
            current_seq=int(data.get("current_seq", 0)),
            applied_seq=int(data.get("applied_seq", 0)),
            index_seq=int(idx_seq) if idx_seq is not None else None,
            catalog_seq=int(data.get("catalog_seq", 0)),
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


# ============================================================================
#  Cluster — GET /cluster
# ============================================================================


@dataclass
class MasterCollectionInfo:
    """Per-collection progress reported by a master."""

    name: str
    catalog_seq: int
    current_seq: int
    applied_seq: int
    index_seq: int | None = None

    @staticmethod
    def from_dict(data: dict) -> "MasterCollectionInfo":
        idx_seq = data.get("index_seq")
        return MasterCollectionInfo(
            name=data.get("name", ""),
            catalog_seq=int(data.get("catalog_seq", 0)),
            current_seq=int(data.get("current_seq", 0)),
            applied_seq=int(data.get("applied_seq", 0)),
            index_seq=int(idx_seq) if idx_seq is not None else None,
        )


@dataclass
class ReplicaCollectionInfo:
    """Per-collection progress reported by a replica."""

    name: str
    applied_catalog_seq: int
    applied_data_seq: int

    @staticmethod
    def from_dict(data: dict) -> "ReplicaCollectionInfo":
        return ReplicaCollectionInfo(
            name=data.get("name", ""),
            applied_catalog_seq=int(data.get("applied_catalog_seq", 0)),
            applied_data_seq=int(data.get("applied_data_seq", 0)),
        )


@dataclass
class ReplicaCursorInfo:
    """One connected replica's last-known cursors, as reported by ACKs."""

    replica_id: str
    applied_global_catalog_seq: int
    collections: list[ReplicaCollectionInfo] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict) -> "ReplicaCursorInfo":
        cols = data.get("collections", [])
        return ReplicaCursorInfo(
            replica_id=data.get("replica_id", ""),
            applied_global_catalog_seq=int(data.get("applied_global_catalog_seq", 0)),
            collections=[
                ReplicaCollectionInfo.from_dict(c) for c in cols if isinstance(c, dict)
            ]
            if isinstance(cols, list)
            else [],
        )


@dataclass
class ClusterInfo:
    """Response payload for ``GET /cluster``.

    The server tags the response with ``role`` (``"standalone"``, ``"master"``,
    or ``"replica"``); the irrelevant fields will be left at their defaults
    (empty lists, ``None``, ``""``, ``0``). Inspect :attr:`role` to know
    which fields are meaningful.
    """

    role: str
    node_id: str
    # master-only
    global_catalog_seq: int = 0
    master_collections: list[MasterCollectionInfo] = field(default_factory=list)
    replicas: list[ReplicaCursorInfo] = field(default_factory=list)
    # replica-only
    master_addr: str = ""
    applied_global_catalog_seq: int = 0
    replica_collections: list[ReplicaCollectionInfo] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict) -> "ClusterInfo":
        role = str(data.get("role", "")).lower()
        info = ClusterInfo(role=role, node_id=str(data.get("node_id", "")))
        if role == "master":
            cols = data.get("collections", [])
            reps = data.get("replicas", [])
            info.global_catalog_seq = int(data.get("global_catalog_seq", 0))
            info.master_collections = (
                [MasterCollectionInfo.from_dict(c) for c in cols if isinstance(c, dict)]
                if isinstance(cols, list)
                else []
            )
            info.replicas = (
                [ReplicaCursorInfo.from_dict(r) for r in reps if isinstance(r, dict)]
                if isinstance(reps, list)
                else []
            )
        elif role == "replica":
            cols = data.get("collections", [])
            info.master_addr = str(data.get("master_addr", ""))
            info.applied_global_catalog_seq = int(
                data.get("applied_global_catalog_seq", 0)
            )
            info.replica_collections = (
                [ReplicaCollectionInfo.from_dict(c) for c in cols if isinstance(c, dict)]
                if isinstance(cols, list)
                else []
            )
        return info


# ============================================================================
#  Configuration — GET /config
# ============================================================================


@dataclass
class AppConfig:
    """Response payload for ``GET /config`` — the runtime configuration the
    server started with. Immutable after startup; restart the server to
    change a value."""

    host: str
    http_port: int
    storage_dir: str
    cluster_role: str
    grpc_port: int
    snapshot_every_writes: int
    snapshot_every_seconds: int
    snapshot_enabled: bool
    sync_replicas: int
    write_ack_timeout_ms: int
    master_addr: str | None = None

    @staticmethod
    def from_dict(data: dict) -> "AppConfig":
        return AppConfig(
            host=str(data.get("host", "")),
            http_port=int(data.get("http_port", 0)),
            storage_dir=str(data.get("storage_dir", "")),
            cluster_role=str(data.get("cluster_role", "")),
            grpc_port=int(data.get("grpc_port", 0)),
            snapshot_every_writes=int(data.get("snapshot_every_writes", 0)),
            snapshot_every_seconds=int(data.get("snapshot_every_seconds", 0)),
            snapshot_enabled=bool(data.get("snapshot_enabled", False)),
            sync_replicas=int(data.get("sync_replicas", 0)),
            write_ack_timeout_ms=int(data.get("write_ack_timeout_ms", 0)),
            master_addr=data.get("master_addr"),
        )
