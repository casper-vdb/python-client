import math
import os
import random
import sys
import time
from pathlib import Path

# Allow running the example without installing the package (editable install).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from casper_client import (
    CasperClient,
    CompactionPolicyPatch,
    IndexPolicyPatch,
    IndexState,
    IndexStorageSpec,
    SnapshotPolicyPatch,
    VectorPoint,
)


def gen_vec(dim: int) -> list[float]:
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def wait_for_index(client: CasperClient, collection: str, *, timeout: float = 60.0) -> None:
    """Poll index.state until READY. NOT_STARTED after a start is a failed build."""
    deadline = time.monotonic() + timeout
    while True:
        state = client.get_collection(collection).index.state
        if state == IndexState.READY:
            return
        if state == IndexState.NOT_STARTED:
            raise RuntimeError(f"index build of {collection!r} failed; see the server log")
        if time.monotonic() > deadline:
            raise TimeoutError(f"index of {collection!r} not ready after {timeout}s")
        time.sleep(0.2)


def main() -> None:
    host = os.getenv("CASPER_HOST", "http://127.0.0.1")
    http_port = int(os.getenv("CASPER_HTTP_PORT", "7222"))
    api_key = os.getenv("CASPER_API_KEY")

    with CasperClient(host=host, http_port=http_port, api_key=api_key) as client:
        # 1) Health / readiness
        client.health()
        print(f"Health: OK, ready: {client.ready()}")

        # 1a) Cluster role / Raft state and one config value
        cluster = client.get_cluster()
        print(f"Cluster: role={cluster.role} node_id={cluster.node_id or '-'}")
        if cluster.role == "cluster":
            print(f"  raft_state={cluster.raft_state} leader_id={cluster.leader_id}")
        config = client.get_config()
        print(f"Config: storage_dir={config.storage.storage_dir}")

        collection = "example_collection"
        dim = 128

        # 2) Create the collection with the schema and policy of its index
        collection_id = client.create_collection(
            collection,
            dim=dim,
            max_size=10_000,
            metric="inner-product",
            quantization="f32",
            m=16,
            m0=32,
            ef_construction=200,
            normalization=True,
            # Optional; every key left out takes the server default.
            storage=IndexStorageSpec(
                snapshot=SnapshotPolicyPatch(changes=1_000),
                compaction=CompactionPolicyPatch(max_tail_ratio=0.25),
            ),
        )
        print(f"Collection created: {collection} ({collection_id})")

        # 3) Write: upserts, then a mixed upsert + delete batch
        points = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(1, 11)]
        client.batch_update(collection, upserts=points, delete=[])
        client.batch_update(
            collection, upserts=[VectorPoint(id=11, vector=gen_vec(dim))], delete=[10]
        )
        print("Upserted vectors 1..11, deleted 10")

        # 4) Start indexing (202; the build is async) and wait for it
        client.start_indexing(collection)
        wait_for_index(client, collection)
        print("Index ready")

        # 5) Change the policy: snapshot more often, turn the chain-depth rule
        # off (None); then snapshot and fold now where the node allows it.
        client.update_index_policy(
            collection,
            IndexPolicyPatch(
                snapshot=SnapshotPolicyPatch(changes=100, wal_ops=200),
                compaction=CompactionPolicyPatch(max_chain_depth=None),
            ),
        )
        if config.storage.index.snapshot_enabled:
            client.trigger_snapshot(collection)
        if config.storage.index.compaction_enabled:
            client.trigger_compaction(collection)
        storage = client.get_collection(collection).index.storage
        print(
            f"Policy: snapshot.changes={storage.snapshot.changes} "
            f"compaction.max_chain_depth={storage.compaction.max_chain_depth}"
        )

        # 6) Search (binary output is default)
        query = gen_vec(dim)
        hits = client.search(collection, query=query, limit=10)
        print(f"Search hits: {len(hits)}")
        for i, h in enumerate(hits[:5], start=1):
            print(f"  {i}. id={h.id} score={h.score}")

        # 7) Scores of the same query against chosen ids (10 is deleted: left out)
        for s in client.scores(collection, query=query, ids=[1, 2, 10]):
            print(f"  score id={s.id} score={s.score}")

        # 8) Get vector by id; None when the index does not hold it
        v1 = client.get_vector(collection, id=1)
        print(f"Vector 1 dim: {len(v1.vector) if v1 else None}, i8: {v1.is_i8 if v1 else None}")
        print(f"Vector 10: {client.get_vector(collection, id=10)}")

        # 9) List collections
        cols = client.list_collections()
        print(f"Collections: {len(cols.collections)}")
        for c in cols.collections:
            info = client.get_collection(c.name)
            print(
                f"  - {info.name} (dim={info.dimension} size={info.size} "
                f"status={info.status} index={info.index.state.value} "
                f"writes={info.wal.write_count})"
            )

        # 10) Cleanup — an index lives until its collection is dropped.
        client.delete_collection(collection)
        print("Cleanup done")


if __name__ == "__main__":
    main()
