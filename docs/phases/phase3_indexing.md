# Phase 3 — Indexing

## Summary
Turns every `Chunk` (Phase 2 output, `chunks.json`) into a hybrid-searchable record in Qdrant. Embeds chunk text into 1024-dim dense vectors via BGE-M3, fits a corpus-wide BM25 model and encodes each chunk into a sparse vector, and upserts both as named vectors on the same point in one Qdrant collection. Output is a Qdrant collection (`financial_rag`) on disk plus a persisted BM25 vocab/idf file — this is what Phase 5 (Retrieval) queries against.

## Worked Example
Continuing the same chunk from Phase 2: `chunk_id=33c5dee5-4d03-4b10-aa04-1f28a2b8ecea`, text starting "...The Company's total net sales were $383.3 billion and net income was $97.0 billion during 2023...", metadata `ticker=AAPL, form=10-K, fiscal_year=2023, item_label=Item 7`.

1. `embedder.embed([chunk.text, ...])` runs this chunk's text (along with the other 736 AAPL chunks in the same batch) through BGE-M3, producing one 1024-dim float vector, L2-normalized so Qdrant's cosine distance is well-behaved. Real vector (first 5 of 1024 dims): `[-0.0657, 0.0003, -0.0088, 0.0266, -0.0326, ...]` — not human-readable on its own, but because this passage shares vocabulary with phrases like "total net sales," "revenue," "fiscal 2023," it lands close in embedding space to other passages discussing the same topic, which is exactly what Phase 5's dense search exploits later.
2. `bm25.fit(all_chunk_texts)` builds a corpus-wide vocabulary (3,643 unique terms across the combined AAPL+MSFT corpus used in this project) and IDF table. Words like `net`, `sales`, `2023` appear in hundreds of chunks across the filing (low IDF, low discriminative weight); a word like `realign` (from "realign the Company's fiscal quarters") appears in only a handful of chunks (high IDF, high weight) — this is exactly why BM25 complements dense search: it rewards rare, exact-term matches that an embedding model might blur together.
3. `bm25.encode_all()` turns this chunk's text into a sparse vector: a list of `(vocab_id, weight)` pairs, one per unique token in the chunk, weight = BM25's term-frequency-with-length-normalization formula. Real result: **52 nonzero terms** out of 3,643 total vocab terms for this one chunk — every other vocab term is implicitly zero for this point.
4. `IndexManager` builds one `PointStruct`: `id="33c5dee5-4d03-4b10-aa04-1f28a2b8ecea"` (the chunk's own UUID, reused directly as the Qdrant point id), `vector={"dense": [1024 floats], "sparse": SparseVector(...)}`, `payload=` the full `Chunk` JSON (text + all `ChunkMetadata` fields) — so a search hit on this point later carries everything Phase 5/6 need with zero extra lookups.
5. `upsert_points()` writes this point (batched with ~99 others) into the `financial_rag` collection at `data/index/qdrant_local/`. `bm25.save_vocab()` writes the fitted vocab+idf to `data/index/bm25_vocab.json` so Phase 5 can encode the query *"What was Apple's net sales in fiscal 2023?"* against this exact same vocabulary later — if the vocab drifted between indexing and querying, sparse search would silently break.

This chunk (one of 1,177 points in the live collection — 737 AAPL + 440 MSFT) is now hybrid-searchable. Phase 5's Worked Example picks up from here with the actual query.

## Files Changed / Added
- `src/indexing/embeddings.py` — modified (stub → full) — `BGEEmbedder`: loads BAAI/bge-m3 once, batched `embed()` returning normalized dense vectors
- `src/indexing/sparse_indexer.py` — modified (stub → full) — `tokenize()`, `BM25SparseEncoder`: fits rank-bm25 over the full corpus, encodes documents/queries into Qdrant `SparseVector`s, persists vocab+idf
- `src/indexing/dense_indexer.py` — modified (stub → full) — `ensure_collection()` (recreate collection with named dense+sparse vectors), `upsert_points()` (batched upsert)
- `src/indexing/index_manager.py` — modified (stub → full) — `IndexManager`: orchestrates embed → BM25 fit/encode → upsert across all chunks in one run
- `scripts/build_index.py` — new — CLI: scans `data/processed/{ticker}/{form}_{year}/chunks.json` → `IndexManager.build_index()` → rich summary table
- `configs/base.yaml` — modified — added `indexing.qdrant_path`, `indexing.bm25_vocab_path`
- `.gitignore` — modified — added `data/index/` (Qdrant local storage + BM25 vocab, regeneratable)
- `.claude/CLAUDE.md` — modified — removed Docker prerequisite, updated "Vector DB" row, updated Running the Project steps, Current Phase
- `tests/unit/test_embeddings.py` — new — `BGEEmbedder.embed()` batching/normalization, fake `SentenceTransformer`
- `tests/unit/test_sparse_indexer.py` — new — tokenize, fit/vocab assignment, BM25 weight correctness (rarer term > weight), query encoding, vocab persistence
- `tests/unit/test_dense_indexer.py` — new — collection schema (named dense+sparse vectors), recreate-wipes-data, batched upsert
- `tests/unit/test_index_manager.py` — new — end-to-end `build_index()` against in-memory Qdrant + fake embedder

## Key Design Decisions
- **No Docker on this machine → Qdrant runs in embedded local mode**: `QdrantClient(path=cfg.indexing.qdrant_path)` persists to `data/index/qdrant_local/` with no server process. Same `qdrant-client` API as server mode — swapping back to `QdrantClient(url="localhost")` later (once Docker is available) is a one-line change, not a rewrite.
- **One Qdrant collection, two named vectors per point** (`dense`, `sparse`): this is the entire reason CLAUDE.md picked Qdrant over a dense-only DB — hybrid search without a second vector store or a join.
- **Collection is recreated fresh every `build_index.py` run, not incrementally upserted**: the corpus is still small and the chunking strategy changes during ablation experiments (`configs/experiments/chunking_ablation.yaml`) — a stale mix of chunks from different strategies in one collection would silently corrupt retrieval comparisons. Revisit once the dataset is large enough that a full rebuild is expensive.
- **BM25 is fit once across ALL chunks passed to `build_index()`, not per filing**: BM25 IDF is only meaningful relative to the whole indexed corpus — fitting per filing would make every term's idf identical within that filing (no discriminative power) and incompatible across filings sharing one collection. This is why `build_index.py` gathers every matching filing's chunks into one list before calling `IndexManager.build_index()` once.
- **BM25 sparse vectors are materialized at index time, not computed at query time**: `rank_bm25.BM25Okapi.get_scores()` does a full corpus scan per query — fine for in-memory toy search, but Qdrant's sparse HNSW index needs each document's BM25 weights as a vector up front so it can do the matching itself.
- **Custom int vocabulary instead of a hashing trick**: each unique token gets a stable, persisted integer id (`bm25_vocab.json`). A hash-based id scheme would avoid persisting a vocab file but risks collisions and isn't reversible for debugging; at this corpus size (~3K terms) persisting the real vocab costs nothing.
- **Document-side BM25 weight vs. query-side IDF-only weight** (`encode_all()` vs. `encode_query()`): documents carry the full BM25 weight (including length normalization via `k1`/`b`/`avgdl`); queries carry only IDF-weighted term presence. Qdrant's sparse dot product between these two reconstructs the BM25 score for matched terms — same convention as Qdrant's own `bm25` fastembed model. Document length normalization belongs on the document side only; doing it twice would double-penalize long documents.
- **Payload stores the full chunk** (`chunk.model_dump(mode="json")`), not just an id pointing back to `chunks.json`: Phase 5 (Retrieval) and Phase 6 (Generation) need chunk text + metadata directly off a search hit — round-tripping to disk per retrieved chunk adds latency for no benefit at this corpus size.
- **`chunk.chunk_id` (a uuid4 string) is reused directly as the Qdrant point id**: Qdrant requires point ids to be either an unsigned int or a valid UUID string — `Chunk.chunk_id` was already a uuid4 from Phase 2's `enricher.py`, so no separate id scheme was needed. (Caught during test-writing: an arbitrary string like `"chunk-0"` is rejected by Qdrant's local-mode point-id validation — tests now use real UUIDs/ints.)

## Execution Flow
1. `python scripts/build_index.py --ticker AAPL --years 2023 --forms 10-K` (or no args → config defaults)
2. `load_config()` resolves `configs/base.yaml`
3. `_find_chunks()` scans `data/processed/{ticker}/{form}_{year}/` for `chunks.json` files (written by Phase 2's `process_filings.py`)
4. every filing's chunks are loaded (`Chunk.model_validate`) and concatenated into one `all_chunks` list
5. `IndexManager(cfg)` constructed: opens embedded `QdrantClient`, loads `BGEEmbedder` (BAAI/bge-m3), creates an unfit `BM25SparseEncoder`
6. `manager.build_index(all_chunks)`:
   a. `ensure_collection()` — deletes the collection if it exists, recreates with `dense` (1024d, cosine, HNSW m=16/ef_construct=200) + `sparse` (BM25) named vectors
   b. `embedder.embed([c.text for c in chunks])` — batched dense embedding, normalized
   c. `bm25.fit(texts)` then `bm25.encode_all()` — corpus-wide BM25 fit + per-chunk sparse vector encoding
   d. `bm25.save_vocab(cfg.indexing.bm25_vocab_path)` — writes `data/index/bm25_vocab.json`
   e. builds one `PointStruct` per chunk: `id=chunk.chunk_id`, `vector={"dense": ..., "sparse": ...}`, `payload=chunk.model_dump(mode="json")`
   f. `upsert_points()` — upserts in batches of 100
7. CLI prints rich summary table (ticker/form/year/chunk count) + logs total points indexed

## Data Contract (Input → Output)
- Input (from Phase 2): `data/processed/{ticker}/{form}_{year}/chunks.json` — `list[Chunk]`
- Output (consumed by Phase 5): Qdrant collection `financial_rag` at `data/index/qdrant_local/` — one point per chunk, named vectors `dense` (1024d) + `sparse` (BM25), payload = full `Chunk` JSON. Plus `data/index/bm25_vocab.json` — `{vocab: {term: int_id}, idf: {term: float}, k1, b, avgdl}`, needed by Phase 5 to encode queries with the same term→id mapping.

## Tests
- `tests/unit/test_embeddings.py` — `embed()` passes `batch_size`/`normalize_embeddings` through to the model, returns a numpy array; `SentenceTransformer` faked, no real model load
- `tests/unit/test_sparse_indexer.py` — tokenization, vocab id assignment, BM25 weight correctness (a term appearing in 1/3 docs outweighs one appearing in 3/3), query encoding drops out-of-vocab terms, vocab+idf persisted to disk
- `tests/unit/test_dense_indexer.py` — collection created with correct dense (size/distance) + sparse named vectors, `ensure_collection()` wipes existing points on recreate, `upsert_points()` batches correctly (forced 3 batches for 5 points at `batch_size=2`); uses `QdrantClient(location=":memory:")`, no mocking of the Qdrant API itself
- `tests/unit/test_index_manager.py` — `build_index()` returns the correct point count, both named vectors land in Qdrant with the right payload, BM25 vocab file is written; `QdrantClient` redirected to in-memory mode, `BGEEmbedder` faked
- Full suite: 48 passed (`python -m pytest tests/unit -q`) — 35 from Phase 1+2, +13 new this phase
- Gap: no test runs the real BGE-M3 model or a real on-disk embedded Qdrant collection (by design); no test exercises `build_index.py`'s CLI/file-finding logic against real `chunks.json` fixtures on disk — covered instead by the manual smoke test below

## Manual Smoke Test (real run, not part of pytest)
`python scripts/build_index.py --ticker AAPL --years 2023 --forms 10-K` against the real 737-chunk AAPL 10-K 2023 corpus from Phase 2:
- 737/737 points indexed, `points_count == 737` confirmed via `client.get_collection()`
- one point inspected directly: `dense` vector len 1024, `sparse` vector with 42 nonzero terms, payload contains all `Chunk` fields
- dense similarity search for "What products does Apple sell?" returned cosine scores ~0.63 against clearly relevant chunks (Item 1 Business description, net sales by product line)
- BM25 vocab persisted: 3,206 unique terms across the corpus

## Config Keys Used
- `configs/base.yaml` → `embeddings.dense_model`, `embeddings.dense_dimensions`, `embeddings.batch_size`, `indexing.collection_name`, `indexing.distance_metric`, `indexing.hnsw_m`, `indexing.hnsw_ef_construct`, `indexing.qdrant_path`, `indexing.bm25_vocab_path`

## Open Items / Deferred
- `is_parent=True` chunks (parent_child chunking strategy) are indexed identically to leaf chunks right now — current corpus only uses `recursive` chunking, so this hasn't mattered yet. Once `chunking.strategy: parent_child` is actually used, `IndexManager.build_index()` should skip parent chunks (they're meant to be looked up by `parent_id` at generation time, not vector-searched) — deferred until that strategy is exercised.
- Incremental upsert (vs. full recreate) deferred until the corpus is large enough that a full rebuild is slow — see Key Design Decisions.
- `src/pipeline.py` still a stub — full end-to-end wiring deferred until Phase 4-6 (query/retrieval/generation) exist.
- `encode_query()` on `BM25SparseEncoder` is implemented but unused until Phase 5 wires up actual query-time sparse retrieval.
