Build the hybrid index (dense + sparse) in Qdrant.

Steps:
1. Verify Qdrant is running at localhost:6333
2. Create (or recreate) the Qdrant collection with both dense and sparse vector configs
3. Generate embeddings for all chunks (batch=64, show progress bar)
4. Upsert all vectors with metadata payloads
5. Report: collection size, indexing time, cost estimate for embeddings
