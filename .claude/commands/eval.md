Run the full evaluation suite against FinanceBench.

Steps:
1. Load FinanceBench dataset from `data/eval/financebench.json`
2. Run retrieval + generation for every question (use batch_size from config)
3. Compute RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
4. Compute retrieval metrics: MRR, NDCG@5, Hit Rate@5
5. Print results table comparing this run vs any prior runs in the experiment log
6. Ask if you want to save these results to the experiment log in CLAUDE.md
