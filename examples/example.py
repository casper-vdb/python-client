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

from casper_client import CasperClient, VectorPoint


def gen_vec(dim: int) -> list[float]:
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    host = os.getenv("CASPER_HOST", "http://127.0.0.1")
    http_port = int(os.getenv("CASPER_HTTP_PORT", "7222"))

    with CasperClient(host=host, http_port=http_port) as client:
        # 1) Health
        client.health()
        print("Health: OK")

        collection = "example_collection"
        dim = 128

        # 2) Create collection
        client.create_collection(collection, dim=dim, max_size=10_000)
        print(f"Collection created: {collection}")

        # 3) Upsert vectors (single batch call replaces per-id insert)
        points = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(1, 6)]
        client.upsert_vectors(collection, points)
        print("Upserted vectors: 1..5")

        # 4) Mixed insert + delete in one call
        batch = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(6, 11)]
        client.batch_update(collection, insert=batch, delete=[])
        print("Batch inserted vectors: 6..10")

        # 5) Create HNSW index (server returns 202; build is async)
        client.create_hnsw_index(
            collection,
            metric="inner-product",
            quantization="i8",
            m=16,
            m0=32,
            ef_construction=200,
            normalization=True,
        )
        print("HNSW index build scheduled")

        # Wait until the index is ready.
        while True:
            info = client.get_collection(collection)
            if info.has_index:
                break
            time.sleep(0.2)
        print("Index ready")

        # 6) Search (binary output is default)
        hits = client.search(collection, query=gen_vec(dim), limit=10)
        print(f"Search hits: {len(hits)}")
        for i, h in enumerate(hits[:5], start=1):
            print(f"  {i}. id={h.id} score={h.score}")

        # 7) Get vector by id
        v1 = client.get_vector(collection, id=1)
        print(f"Vector 1 dim: {len(v1.vector)}")

        # 8) Delete vectors by id (batch)
        client.delete_vectors(collection, [10])
        print("Vector 10 deleted")

        # 9) Mute / unmute roundtrip
        client.mute_collection(collection)
        client.unmute_collection(collection)
        print("Mute/unmute roundtrip ok")

        # 10) List collections
        cols = client.list_collections()
        print(f"Collections: {len(cols.collections)}")
        for c in cols.collections:
            print(
                f"  - {c.name} (dim={c.dimension} size={c.size} "
                f"status={c.status} mutable={c.mutable} has_index={c.has_index})"
            )

        # 11) Cleanup
        client.delete_index(collection)
        client.delete_collection(collection)
        print("Cleanup done")


if __name__ == "__main__":
    main()
