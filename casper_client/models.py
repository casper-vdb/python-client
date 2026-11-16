from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
class SnapshotPolicyInfo:
    enabled: bool
    # Changed ids since the last snapshot that trigger the next one.
    changes: int
    # Backstop: WAL ops since the last snapshot.
    wal_ops: int

    @staticmethod
    def from_dict(data: dict) -> "SnapshotPolicyInfo":
        return SnapshotPolicyInfo(
            enabled=bool(data.get("enabled", False)),
            changes=int(data.get("changes", 0)),
            wal_ops=int(data.get("wal_ops", 0)),
        )


@dataclass
class CompactionPolicyInfo:
    enabled: bool
    # None: the chain-depth rule is off.
    max_chain_depth: int | None
    max_tail_ratio: float

    @staticmethod
    def from_dict(data: dict) -> "CompactionPolicyInfo":
        depth = data.get("max_chain_depth")
        return CompactionPolicyInfo(
            enabled=bool(data.get("enabled", False)),
            max_chain_depth=int(depth) if depth is not None else None,
            max_tail_ratio=float(data.get("max_tail_ratio", 0.0)),
        )


@dataclass
class IndexStorageInfo:
    range_size_bytes: int
    snapshot: SnapshotPolicyInfo
    compaction: CompactionPolicyInfo
    # The node's flag and the policy's flag together.
    snapshot_active: bool
    compaction_active: bool

    @staticmethod
    def from_dict(data: dict) -> "IndexStorageInfo":
        snapshot = data.get("snapshot")
        compaction = data.get("compaction")
        return IndexStorageInfo(
            range_size_bytes=int(data.get("range_size_bytes", 0)),
            snapshot=SnapshotPolicyInfo.from_dict(snapshot if isinstance(snapshot, dict) else {}),
            compaction=CompactionPolicyInfo.from_dict(
                compaction if isinstance(compaction, dict) else {}
            ),
            snapshot_active=bool(data.get("snapshot_active", False)),
            compaction_active=bool(data.get("compaction_active", False)),
        )


class IndexState(str, Enum):
    """``index.state`` of ``GET /collections/{name}``."""

    # Indexing has not been started; after a start, a build that failed
    # (standalone).
    NOT_STARTED = "not_started"
    # A build is running, or in a cluster this node has not fetched the index yet.
    BUILDING = "building"
    # An index is attached.
    READY = "ready"


@dataclass
class IndexInfo:
    """The schema and policy the collection was created with, and where
    indexing is."""

    state: IndexState
    hnsw: HnswIndexInfo
    normalization: bool
    storage: IndexStorageInfo

    @staticmethod
    def from_dict(data: dict) -> "IndexInfo":
        return IndexInfo(
            state=IndexState(str(data.get("state", IndexState.NOT_STARTED.value))),
            hnsw=HnswIndexInfo.from_dict(_section(data, "hnsw")),
            normalization=bool(data.get("normalization", False)),
            storage=IndexStorageInfo.from_dict(_section(data, "storage")),
        )


@dataclass
class WalCounters:
    """Writes (one WAL record each) counted from the collection's creation.
    Without an index the last two are 0."""

    # Writes durably in the WAL.
    write_count: int
    # Writes applied to the in-memory index, so visible to search.
    applied_count: int
    # Writes the index on disk covers; `write_count - snapshotted_count` is
    # what a restart replays.
    snapshotted_count: int

    @staticmethod
    def from_dict(data: dict) -> "WalCounters":
        return WalCounters(
            write_count=int(data.get("write_count", 0)),
            applied_count=int(data.get("applied_count", 0)),
            snapshotted_count=int(data.get("snapshotted_count", 0)),
        )


@dataclass
class SnapshotCounters:
    """What the snapshot thresholds are compared against, since the last snapshot."""

    changes: int
    wal_ops: int

    @staticmethod
    def from_dict(data: dict) -> "SnapshotCounters":
        return SnapshotCounters(
            changes=int(data.get("changes", 0)),
            wal_ops=int(data.get("wal_ops", 0)),
        )


@dataclass
class CollectionInfo:
    """Response payload for ``GET /collections/{name}``."""

    id: str
    name: str
    dimension: int
    has_index: bool
    max_size: int
    # Documents durably stored.
    size: int
    # "yellow" while an index is being built, else "green".
    status: str
    # Always there: the schema and policy the collection was created with.
    index: IndexInfo
    wal: WalCounters
    # Documents visible to search; lags `size` by the applier backlog.
    applied_size: int = 0
    # The collection was created in cluster mode.
    replicated: bool = False
    # Counters since the last snapshot; None while no index is attached.
    snapshot: SnapshotCounters | None = None

    @staticmethod
    def from_dict(data: dict) -> "CollectionInfo":
        snapshot = data.get("snapshot")
        return CollectionInfo(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            dimension=int(data.get("dimension", 0)),
            has_index=bool(data.get("has_index", False)),
            max_size=int(data.get("max_size", 0)),
            size=int(data.get("size", 0)),
            status=str(data.get("status", "")),
            index=IndexInfo.from_dict(_section(data, "index")),
            wal=WalCounters.from_dict(_section(data, "wal")),
            applied_size=int(data.get("applied_size", 0)),
            replicated=bool(data.get("replicated", False)),
            snapshot=SnapshotCounters.from_dict(snapshot) if isinstance(snapshot, dict) else None,
        )


@dataclass
class CollectionDescription:
    """One entry of ``GET /collections``; use ``get_collection`` for details."""

    name: str

    @staticmethod
    def from_dict(data: dict) -> "CollectionDescription":
        return CollectionDescription(name=str(data.get("name", "")))


@dataclass
class CollectionsListResponse:
    collections: list[CollectionDescription] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict) -> "CollectionsListResponse":
        cols = data.get("collections", [])
        if not isinstance(cols, list):
            cols = []
        return CollectionsListResponse(
            collections=[CollectionDescription.from_dict(c) for c in cols if isinstance(c, dict)]
        )


# ============================================================================
#  Index policy — POST /collection/{name}, PATCH /collection/{name}/index
# ============================================================================


class _Unset:
    """Type of :data:`UNSET`: a key left out of the request."""

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


# Leaves a key out of the request where ``None`` already means something:
# ``CompactionPolicyPatch.max_chain_depth=None`` turns the depth rule off.
UNSET = _Unset()


@dataclass
class SnapshotPolicyPatch:
    """Snapshot policy keys to set; ``None`` leaves a key as it is."""

    enabled: bool | None = None
    # Changed ids since the last snapshot that trigger the next one (>= 1).
    changes: int | None = None
    # Backstop: WAL ops since the last snapshot (>= changes).
    wal_ops: int | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        if self.enabled is not None:
            out["enabled"] = bool(self.enabled)
        if self.changes is not None:
            out["changes"] = int(self.changes)
        if self.wal_ops is not None:
            out["wal_ops"] = int(self.wal_ops)
        return out


@dataclass
class CompactionPolicyPatch:
    """Compaction policy keys to set; ``None`` leaves a key as it is, except
    ``max_chain_depth``: :data:`UNSET` leaves it, ``None`` turns the depth rule
    off, an int (>= 3) sets it."""

    enabled: bool | None = None
    max_chain_depth: int | None | _Unset = UNSET
    # In (0, 1).
    max_tail_ratio: float | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        if self.enabled is not None:
            out["enabled"] = bool(self.enabled)
        if self.max_chain_depth is None:
            out["max_chain_depth"] = None
        elif not isinstance(self.max_chain_depth, _Unset):
            out["max_chain_depth"] = int(self.max_chain_depth)
        if self.max_tail_ratio is not None:
            out["max_tail_ratio"] = float(self.max_tail_ratio)
        return out


@dataclass
class IndexPolicyPatch:
    """Body of ``PATCH /collection/{name}/index``: only the named keys change."""

    snapshot: SnapshotPolicyPatch | None = None
    compaction: CompactionPolicyPatch | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        if self.snapshot is not None:
            out["snapshot"] = self.snapshot.to_dict()
        if self.compaction is not None:
            out["compaction"] = self.compaction.to_dict()
        return out


@dataclass
class IndexStorageSpec:
    """``index.storage`` of ``POST /collection/{name}``: what is left out takes
    the server default."""

    # Width of a payload range on disk, 128 MiB..8 GiB; fixed for the life of
    # the index.
    range_size_bytes: int | None = None
    snapshot: SnapshotPolicyPatch | None = None
    compaction: CompactionPolicyPatch | None = None

    def to_dict(self) -> dict:
        out: dict = {}
        if self.range_size_bytes is not None:
            out["range_size_bytes"] = int(self.range_size_bytes)
        if self.snapshot is not None:
            out["snapshot"] = self.snapshot.to_dict()
        if self.compaction is not None:
            out["compaction"] = self.compaction.to_dict()
        return out


@dataclass
class SearchResult:
    """One hit of ``search`` or ``scores``; ``id`` is a u64 point id."""

    id: int
    score: float


@dataclass
class VectorPoint:
    """A single (id, vector) pair to upsert; ``id`` is a u64 point id."""

    id: int
    vector: list[float]

    def to_dict(self) -> dict:
        return {"id": int(self.id), "vector": list(self.vector)}


@dataclass
class GetVectorResponse:
    """The vector exactly as the index holds it. An f32 index answers floats
    (normalized, if the index normalizes); an i8 index answers its quantized
    integers in [-100, 100], ``round(x * 100)``, not dequantized — then
    :attr:`is_i8` is True."""

    id: int
    vector: list[float] | list[int]
    is_i8: bool = False

    @staticmethod
    def from_dict(data: dict) -> "GetVectorResponse":
        vec = data.get("vector")
        components = list(vec) if isinstance(vec, list) else []
        # The server prints an f32 component with a fraction ("1.0"), so JSON
        # integers are the i8 form.
        is_i8 = bool(components) and all(
            isinstance(c, int) and not isinstance(c, bool) for c in components
        )
        return GetVectorResponse(
            id=int(data.get("id", 0)),
            vector=[int(c) for c in components] if is_i8 else [float(c) for c in components],
            is_i8=is_i8,
        )


# ============================================================================
#  Cluster — GET /cluster
# ============================================================================


@dataclass
class ClusterMemberInfo:
    """One member of the Raft membership, with both addresses it published."""

    node_id: str
    # Where peers dial it for consensus.
    raft_addr: str
    # Where peers dial it for data-WAL replication.
    replication_addr: str
    voter: bool

    @staticmethod
    def from_dict(data: dict) -> "ClusterMemberInfo":
        return ClusterMemberInfo(
            node_id=str(data.get("node_id", "")),
            raft_addr=str(data.get("raft_addr", "")),
            replication_addr=str(data.get("replication_addr", "")),
            voter=bool(data.get("voter", False)),
        )


@dataclass
class ClusterInfo:
    """Response payload for ``GET /cluster``.

    The server tags the response with ``role`` (``"standalone"`` or
    ``"cluster"``). A standalone node has no cluster identity and reports
    only its role; the other fields are left at their defaults (``None``,
    ``""``, ``0``, empty lists).
    """

    role: str
    # None on a standalone node.
    node_id: str | None = None
    # One of leader, follower, candidate, learner, shutdown.
    raft_state: str = ""
    current_term: int = 0
    # None while no leader is known.
    leader_id: str | None = None
    voters: list[str] = field(default_factory=list)
    learners: list[str] = field(default_factory=list)
    members: list[ClusterMemberInfo] = field(default_factory=list)
    last_log_index: int | None = None
    last_applied_index: int | None = None

    @staticmethod
    def from_dict(data: dict) -> "ClusterInfo":
        role = str(data.get("role", "")).lower()
        info = ClusterInfo(role=role)
        if role == "cluster":
            info.node_id = str(data.get("node_id", ""))
            voters = data.get("voters", [])
            learners = data.get("learners", [])
            members = data.get("members", [])
            leader_id = data.get("leader_id")
            last_log = data.get("last_log_index")
            last_applied = data.get("last_applied_index")
            info.raft_state = str(data.get("raft_state", ""))
            info.current_term = int(data.get("current_term", 0))
            info.leader_id = str(leader_id) if leader_id is not None else None
            info.voters = [str(v) for v in voters] if isinstance(voters, list) else []
            info.learners = [str(v) for v in learners] if isinstance(learners, list) else []
            info.members = (
                [ClusterMemberInfo.from_dict(m) for m in members if isinstance(m, dict)]
                if isinstance(members, list)
                else []
            )
            info.last_log_index = int(last_log) if last_log is not None else None
            info.last_applied_index = int(last_applied) if last_applied is not None else None
        return info


# ============================================================================
#  Configuration — GET /config
# ============================================================================


def _section(data: dict, key: str) -> dict:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


@dataclass
class ServiceConfig:
    host: str
    http_port: int
    # The address the node publishes; None means derived from host and port.
    uri: str | None
    shutdown_timeout_sec: int

    @staticmethod
    def from_dict(data: dict) -> "ServiceConfig":
        return ServiceConfig(
            host=str(data.get("host", "")),
            http_port=int(data.get("http_port", 0)),
            uri=_optional_str(data.get("uri")),
            shutdown_timeout_sec=int(data.get("shutdown_timeout_sec", 0)),
        )


@dataclass
class RaftConfig:
    port: int
    enable_tls: bool
    uri: str | None

    @staticmethod
    def from_dict(data: dict) -> "RaftConfig":
        return RaftConfig(
            port=int(data.get("port", 0)),
            enable_tls=bool(data.get("enable_tls", False)),
            uri=_optional_str(data.get("uri")),
        )


@dataclass
class ReplicationConfig:
    port: int
    uri: str | None
    max_transfer_mb_per_sec: int

    @staticmethod
    def from_dict(data: dict) -> "ReplicationConfig":
        return ReplicationConfig(
            port=int(data.get("port", 0)),
            uri=_optional_str(data.get("uri")),
            max_transfer_mb_per_sec=int(data.get("max_transfer_mb_per_sec", 0)),
        )


@dataclass
class ClusterConfig:
    # "standalone" or "cluster".
    role: str
    bootstrap: str | None
    write_ack_timeout_ms: int
    max_inflight_entries: int
    max_apply_lag_mb: int
    write_queue_mb: int
    raft: RaftConfig
    replication: ReplicationConfig

    @staticmethod
    def from_dict(data: dict) -> "ClusterConfig":
        return ClusterConfig(
            role=str(data.get("role", "")),
            bootstrap=_optional_str(data.get("bootstrap")),
            write_ack_timeout_ms=int(data.get("write_ack_timeout_ms", 0)),
            max_inflight_entries=int(data.get("max_inflight_entries", 0)),
            max_apply_lag_mb=int(data.get("max_apply_lag_mb", 0)),
            write_queue_mb=int(data.get("write_queue_mb", 0)),
            raft=RaftConfig.from_dict(_section(data, "raft")),
            replication=ReplicationConfig.from_dict(_section(data, "replication")),
        )


@dataclass
class WalConfig:
    max_batch_size_mb: int
    log_tail_mb: int
    snapshot_every_mb: int
    snapshot_every_entries: int

    @staticmethod
    def from_dict(data: dict) -> "WalConfig":
        return WalConfig(
            max_batch_size_mb=int(data.get("max_batch_size_mb", 0)),
            log_tail_mb=int(data.get("log_tail_mb", 0)),
            snapshot_every_mb=int(data.get("snapshot_every_mb", 0)),
            snapshot_every_entries=int(data.get("snapshot_every_entries", 0)),
        )


@dataclass
class IndexSwitchesConfig:
    """``storage.index``: the node's switches only; the thresholds are each
    index's own, in ``get_collection``."""

    snapshot_enabled: bool
    compaction_enabled: bool

    @staticmethod
    def from_dict(data: dict) -> "IndexSwitchesConfig":
        return IndexSwitchesConfig(
            snapshot_enabled=bool(_section(data, "snapshot").get("enabled", False)),
            compaction_enabled=bool(_section(data, "compaction").get("enabled", False)),
        )


@dataclass
class StorageConfig:
    storage_dir: str
    min_free_disk_mb: int
    wal: WalConfig
    index: IndexSwitchesConfig

    @staticmethod
    def from_dict(data: dict) -> "StorageConfig":
        return StorageConfig(
            storage_dir=str(data.get("storage_dir", "")),
            min_free_disk_mb=int(data.get("min_free_disk_mb", 0)),
            wal=WalConfig.from_dict(_section(data, "wal")),
            index=IndexSwitchesConfig.from_dict(_section(data, "index")),
        )


@dataclass
class SearchConfig:
    # Highest `limit` a search takes.
    max_search_limit: int
    # Most ids one `scores` request takes.
    max_scores_ids: int

    @staticmethod
    def from_dict(data: dict) -> "SearchConfig":
        return SearchConfig(
            max_search_limit=int(data.get("max_search_limit", 0)),
            max_scores_ids=int(data.get("max_scores_ids", 0)),
        )


@dataclass
class TlsConfig:
    enable: bool
    verify_https_client_certificate: bool
    tls_cert_ttl: int | None

    @staticmethod
    def from_dict(data: dict) -> "TlsConfig":
        ttl = data.get("tls_cert_ttl")
        return TlsConfig(
            enable=bool(data.get("enable", False)),
            verify_https_client_certificate=bool(
                data.get("verify_https_client_certificate", False)
            ),
            tls_cert_ttl=int(ttl) if ttl is not None else None,
        )


@dataclass
class AuthConfig:
    jwt_rbac: bool

    @staticmethod
    def from_dict(data: dict) -> "AuthConfig":
        return AuthConfig(jwt_rbac=bool(data.get("jwt_rbac", False)))


@dataclass
class MetricsConfig:
    prefix: str

    @staticmethod
    def from_dict(data: dict) -> "MetricsConfig":
        return MetricsConfig(prefix=str(data.get("prefix", "")))


@dataclass
class AppConfig:
    """Response payload for ``GET /config`` — the configuration the node
    started with, grouped the way the config file is. Immutable after
    startup; restart the server to change a value. Key material and
    credentials are never included."""

    service: ServiceConfig
    cluster: ClusterConfig
    storage: StorageConfig
    search: SearchConfig
    tls: TlsConfig
    auth: AuthConfig
    metrics: MetricsConfig

    @staticmethod
    def from_dict(data: dict) -> "AppConfig":
        return AppConfig(
            service=ServiceConfig.from_dict(_section(data, "service")),
            cluster=ClusterConfig.from_dict(_section(data, "cluster")),
            storage=StorageConfig.from_dict(_section(data, "storage")),
            search=SearchConfig.from_dict(_section(data, "search")),
            tls=TlsConfig.from_dict(_section(data, "tls")),
            auth=AuthConfig.from_dict(_section(data, "auth")),
            metrics=MetricsConfig.from_dict(_section(data, "metrics")),
        )
