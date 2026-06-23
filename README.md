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

from casper_client import CasperClient, VectorPoint


def gen_vec(dim: int) -> list[float]:
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def main() -> None:
    host = os.getenv("CASPER_HOST", "http://127.0.0.1")
    http_port = int(os.getenv("CASPER_HTTP_PORT", "7222"))

    with CasperClient(host=host, http_port=http_port) as client:
        client.health()

        collection = "example_collection"
        dim = 128

        client.create_collection(collection, dim=dim, max_size=10_000)

        # Upsert vectors (batch)
        points = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(1, 6)]
        client.upsert_vectors(collection, points)

        # Mixed insert + delete
        batch = [VectorPoint(id=vid, vector=gen_vec(dim)) for vid in range(6, 11)]
        client.batch_update(collection, insert=batch, delete=[])

        # Create HNSW index — build is asynchronous on the server.
        client.create_hnsw_index(
            collection,
            metric="inner-product",
            quantization="i8",
            m=16,
            m0=32,
            ef_construction=200,
            normalization=True,
        )

        # Wait until the index is ready before searching.
        while not client.get_collection(collection).has_index:
            time.sleep(0.2)

        hits = client.search(collection, query=gen_vec(dim), limit=10)
        for i, h in enumerate(hits[:5], start=1):
            print(f"  {i}. id={h.id} score={h.score}")

        client.delete_vectors(collection, [10])
        client.delete_index(collection)
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
- Collection management (create, delete, list, info)
- Vector operations: `upsert_vectors` (batch PUT), `delete_vectors` (batch by id), `batch_update` (mixed insert+delete), `get_vector`
- Mute / unmute (`mute_collection`, `unmute_collection`)
- Index management (`create_hnsw_index`, `delete_index`) — the index build is asynchronous; the server returns 202 and the client should poll `get_collection().has_index`
- Search (`search`, with optional `ef_search` override). Binary response format is decoded automatically.

## License

Licensed under the [Apache License Version 2.0](LICENSE).
