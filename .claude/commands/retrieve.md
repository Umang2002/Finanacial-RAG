Test the retrieval pipeline interactively with a query.

Steps:
1. Take a query string as input
2. Run dense retrieval → show top-5 results with scores
3. Run sparse (BM25) retrieval → show top-5 results with scores
4. Run hybrid (RRF) retrieval → show top-5 fused results
5. Run reranker on hybrid results → show final top-5
6. Print a comparison table: which docs appeared in dense-only, sparse-only, both
7. Generate and print the final answer with citations

This lets you inspect and debug each retrieval stage independently.
