import os
import random
import math
import sys
from pathlib import Path

# Allow running the example without installing the package (editable install).
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from casper_client import CasperClient


def gen_vec(dim: int) -> list[float]:
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    host = os.getenv("CASPER_HOST", "http://127.0.0.1")
    http_port = int(os.getenv("CASPER_HTTP_PORT", "8080"))
    grpc_port = int(os.getenv("CASPER_GRPC_PORT", "50051"))

    client = CasperClient(host=host, http_port=http_port, grpc_port=grpc_port)

    # 1) Health
    client.health()
    print("Health: OK")

    collection = "example_collection"
    dim = 128

    # 2) Create collection
    client.create_collection(collection, dim=dim, max_size=10_000)
    print(f"Collection created: {collection}")

    # 3) Insert vectors
    for vid in range(1, 6):
        client.insert_vector(collection, id=vid, vector=gen_vec(dim))
    print("Inserted vectors: 1..5")

    # 4) Batch insert more vectors
    batch = [{"id": vid, "vector": gen_vec(dim)} for vid in range(6, 11)]
    client.batch_update(collection, insert=batch, delete=[])
    print("Batch inserted vectors: 6..10")

    # 5) Create HNSW index
    client.create_hnsw_index(
        collection,
        metric="inner-product",
        quantization="i8",
        m=16,
        m0=32,
        ef_construction=200,
        normalization=True,
    )
    print("HNSW index created")

    # 6) Search (binary output is default)
    hits = client.search(collection, query=gen_vec(dim), limit=10)
    print(f"Search hits: {len(hits)}")
    for i, h in enumerate(hits[:5], start=1):
        print(f"  {i}. id={h.id} score={h.score}")

    # 7) Get vector by id
    v1 = client.get_vector(collection, id=1)
    print(f"Vector 1 dim: {len(v1.vector)}")

    # 8) Delete a vector
    client.delete_vector(collection, id=10)
    print("Vector 10 deleted")

    # 9) List collections
    cols = client.list_collections()
    print(f"Collections: {len(cols.collections)}")

    # 10) Cleanup
    client.delete_index(collection)
    client.delete_collection(collection)
    print("Cleanup done")

    client.close()


if __name__ == "__main__":
    main()


