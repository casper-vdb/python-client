# Casper Python client

Python client library for the [Casper Vector Database](https://github.com/casper-vdb/Casper).

## Installation

```bash
pip install casper-client
```

## Quick start

```python
import math
import os
import random
import time

from casper_client import (
    CasperClient,
    CompactionPolicyPatch,
    IndexPolicyPatch,
    IndexState,
    SnapshotPolicyPatch,
    VectorPoint,
)


def gen_vec(dim: int) -> list[float]:
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    host = os.getenv("CASPER_HOST", "http://127.0.0.1")
    http_port = int(os.getenv("CASPER_HTTP_PORT", "7222"))
    api_key = os.getenv("CASPER_API_KEY")  # only if the server has API keys configured

    with CasperClient(host=host, http_port=http_port, api_key=api_key) as client:
        client.health()

        collection = "example_collection"
        dim = 128

        # A collection is created with the schema of its index; returns its id.
        client.create_collection(
            collection,
            dim=dim,
            max_size=10_000,
            metric="inner-product",
            quantization="f32",
            m=16,
            m0=32,
            ef_construction=200,
            normalization=True,
        )

        # The one write: upserts and deletes in one batch.
        points = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(1, 11)]
        client.batch_update(collection, upserts=points, delete=[])
        client.batch_update(collection, upserts=[], delete=[10])

        # Start indexing — the build is asynchronous on the server. Wait on
        # index.state; NOT_STARTED after a start means the build failed.
        client.start_indexing(collection)
        deadline = time.monotonic() + 60
        while (state := client.get_collection(collection).index.state) != IndexState.READY:
            if state == IndexState.NOT_STARTED or time.monotonic() > deadline:
                raise RuntimeError(f"index not ready: {state.value}")
            time.sleep(0.2)

        # Tune the index: snapshot more often, turn the chain-depth rule off
        # (None; leave the field out to keep it as it is).
        client.update_index_policy(
            collection,
            IndexPolicyPatch(
                snapshot=SnapshotPolicyPatch(changes=100),
                compaction=CompactionPolicyPatch(max_chain_depth=None),
            ),
        )

        query = gen_vec(dim)
        hits = client.search(collection, query=query, limit=10)
        for i, h in enumerate(hits[:5], start=1):
            print(f"  {i}. id={h.id} score={h.score}")

        # Scores of the query against chosen ids; ids the index lacks are left out.
        print(client.scores(collection, query=query, ids=[1, 2, 10]))

        # None when the index does not hold the id.
        v = client.get_vector(collection, id=1)
        print(v.vector[:4] if v else "not in the index")

        # An index lives until its collection is dropped.
        client.delete_collection(collection)


if __name__ == "__main__":
    main()
```

## Examples

Run the example provided in this repository:

```bash
python examples/example.py
```

The example demonstrates:
- Health and readiness (`health`, `ready`, with `quorum=True` for rolling restarts)
- Cluster and node info: `get_cluster` (role, Raft state, membership), `get_config` (startup configuration)
- Collection management: `create_collection` with the index schema and optional `storage=IndexStorageSpec(...)` (returns the collection id), `delete_collection`, `list_collections` (names), `get_collection` (info, WAL counters, `index.state`)
- Writes: `batch_update` (upserts + deletes in one batch, u64 ids)
- Indexing: `start_indexing` — the build is asynchronous; the server returns 202 and the client polls `get_collection().index.state` until `READY` (`NOT_STARTED` again means the build failed). An index lives until its collection is dropped
- Index policy: `update_index_policy` (snapshot / compaction patch), `trigger_snapshot`, `trigger_compaction` (guarded by the node's switches in `get_config().storage.index`)
- Reads: `search` (with optional `ef_search`), `scores` (the query against chosen ids), `get_vector` (`None` if the index does not hold the id; an i8 index answers its quantized integers, `is_i8=True`). The binary response format is decoded automatically.

The client also covers the operator endpoints, which the example does not call: `remove_peer` (drop a node from
the cluster membership), `set_transfer_limit` (rate limit of index and WAL transfers) and `get_metrics`
(Prometheus text).

Pass `api_key=` to `CasperClient` when the server has API keys configured; it is sent as the `api-key` header.
Errors are raised as `CasperError`, whose `kind` names the failure. A mutation sent to a cluster node that is
not the leader raises `kind=NOT_LEADER` with `leader_addr` set when a leader is known; 429 and 503 carry
`retry_after` (seconds). A request that got no answer (connection refused, timeout) raises
`kind=TRANSPORT` with the httpx error in `cause`. The client does not retry on its own.

## License

Licensed under the [Apache License Version 2.0](LICENSE).
