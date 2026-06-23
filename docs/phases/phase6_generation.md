# Phase 6 — Generation

## Summary
Turns Phase 5's reranked chunks into a final, citation-grounded answer. `ContextAssembler` packs chunks into a prompt block (lost-in-middle-aware ordering, `[n]` citation tags), `Generator` sends a citation-aware system prompt + that context to the local Ollama LLM, and `output_parser` extracts the clean answer text, the citations actually referenced, and a self-reported confidence score into a `GeneratedAnswer`. No state left on disk — pure library code called per-query.

## Worked Example
Continuing with Phase 5's real top-5 reranked chunks (Item 8, Item 8, Table 19, Item 7, Item 7 — see Phase 5's Worked Example for the full list) for query *"What was Apple's net sales in fiscal 2023?"*. (As in Phase 4's doc: this originally ran on local Ollama, current real output below is post-migration via `GroqClient` — same interface, different backend.)

1. `ContextAssembler.assemble(chunks)` assigns citation numbers `[1]`-`[5]` in the rerank order above (1 = best), then physically reorders the *text* into a sandwich: best chunks ([1], [2]) at both ends of the context block, weakest ([3]) in the middle — so citation numbers stay meaningful to the model while mitigating lost-in-middle attention decay. Produces one `AssembledContext.text` block like:
   ```
   [1] ...The U.S. and China were the only countries that accounted for more
   than 10% of the Company's net sales in 2023...
   [2] ...Net sales disaggregated by significant products and services for
   2023, 2022 and 2021...
   [4] ...iPhone net sales decreased...
   [3] iPhone (1) (2023): $. iPhone (1) (): 200,583...
   [5] ...Products and Services Performance The following table shows net
   sales by category...
   ```
2. `Generator.generate()` sends the citation-aware system prompt + this context + the question to the LLM. Real response (before parsing):
   ```
   Apple's net sales in fiscal 2023 were $383,285 million [1], [2], [3], [5].
   Confidence: 1.0
   ```
3. `parse_llm_output()` strips the trailing `Confidence: 1.0` line, extracts the `[1]`, `[2]`, `[3]`, `[5]` markers (note: `[4]` was in context but never cited — correctly dropped, since an uncited chunk isn't evidence for anything actually said), and resolves each back to its source chunk via `ContextAssembler`'s citation list. Final real `GeneratedAnswer`:
   ```python
   GeneratedAnswer(
       query="What was Apple's net sales in fiscal 2023?",
       answer="Apple's net sales in fiscal 2023 were $383,285 million [1], [2], [3], [5].",
       confidence=1.0,
       citations=[
           Citation(citation_id=1, chunk_id="e73383c0-...", ticker="AAPL", form="10-K", fiscal_year=2023, item_label="Item 8"),
           Citation(citation_id=2, chunk_id="5baad8f8-...", ticker="AAPL", form="10-K", fiscal_year=2023, item_label="Item 8"),
           Citation(citation_id=3, chunk_id="c45a378b-...", ticker="AAPL", form="10-K", fiscal_year=2023, item_label="Table 19"),
           Citation(citation_id=5, chunk_id="0a0455b5-...", ticker="AAPL", form="10-K", fiscal_year=2023, item_label="Item 7"),
       ],
   )
   ```
   `$383,285 million` is the exact figure from the real 10-K — correct, and the citations genuinely trace back to the chunks that state it (Item 8's financial statements, Table 19's category breakdown). This is the full Phase 1-6 pipeline working end to end on one real question.

## Files Changed / Added
- `src/generation/models.py` — new — `Citation`, `GeneratedAnswer`, `AssembledContext` pydantic models
- `src/generation/context_assembler.py` — modified (stub → full) — `ContextAssembler.assemble()`: relevance-ordered citation numbering + lost-in-middle sandwich reordering of chunk text
- `src/generation/output_parser.py` — modified (stub → full) — `parse_llm_output()`: regex-extracts `[n]` citation refs and a trailing `Confidence: 0.8` line
- `src/generation/generator.py` — modified (stub → full) — `Generator.generate()`: wires assembler + `OllamaClient` + parser into one call
- `scripts/generate.py` — new — first full Phase 1-6 end-to-end CLI
- `.claude/CLAUDE.md` — modified — Current Phase bumped to 6, Phase 6 notes added
- `tests/unit/test_context_assembler.py`, `test_output_parser.py`, `test_generator.py` — new

## Key Design Decisions
- **Citation numbers reflect relevance rank; physical placement in the prompt does not**: numbers are assigned 1..N in the order Phase 5 ranked chunks (1 = most relevant), but the text itself is reordered into a "sandwich" (best chunks at both ends of the context block, weakest in the middle) before being sent to the LLM. This decouples "what rank does the model see" from "where in the context window does attention degrade" — the whole point of lost-in-middle mitigation.
- **`GeneratedAnswer.citations` only includes citations the answer actually references**, not every chunk handed to the LLM — a chunk that sat in context but was never cited isn't evidence for anything the model said. `output_parser` cross-references `[n]` markers in the raw text against `ContextAssembler`'s full citation list and keeps only the matches.
- **`citation_style` is config-gated but only `inline` is implemented**: `configs/base.yaml` declares `footnote`/`list` as future options from an earlier design pass, but no prompt template or parser exists for them. `Generator.__init__` raises `ValueError` on anything but `inline` rather than silently producing output `output_parser` can't parse.
- **Confidence is parsed from a structured trailing line (`Confidence: 0.8`), not asked for as JSON**: local 3B models are unreliable at strict JSON output; a single regex-matchable line after free-text prose is far more robust, at the cost of one extra prompt instruction.
- **Unanswerable questions aren't special-cased in code**: the system prompt instructs the LLM to say so in plain text rather than guess; `output_parser` then naturally produces zero citations for that answer (no `[n]` markers to extract).

## Execution Flow
One real run via `scripts/generate.py --query "..."`:
1. `load_config()` resolves `configs/base.yaml`
2. `QueryAnalyzer(cfg).analyze(query)` → intent + `is_multi_hop` flag
3. `QueryTransformer(cfg).transform(query)` → HyDE doc + multi-query paraphrases (per `cfg.query.*` flags)
4. `QueryDecomposer(cfg).decompose(query)` → only called if `is_multi_hop`
5. `RetrievalPipeline(cfg).retrieve(query, transformed, decomposed)` → Phase 5's reranked `list[RetrievedChunk]`
6. `Generator(cfg).generate(query, chunks)`:
   a. `ContextAssembler.assemble(chunks)` → citation-tagged context text + `Citation` list
   b. `OllamaClient.complete()` with the citation-aware system prompt + assembled context + query
   c. `parse_llm_output()` → clean answer, referenced citations, confidence
7. CLI prints the answer, confidence, and a citations table (ticker/form/year/section per `[n]`)

## Data Contract (Input → Output)
- Input (from Phase 5): `list[RetrievedChunk]`, best-first
- Output (consumed by Phase 7 Evaluation): `GeneratedAnswer` — `query: str`, `answer: str`, `citations: list[Citation]`, `confidence: float | None`

## Tests
- `test_context_assembler.py` — sandwich reorder math, relevance-ordered citation assignment, bracketed text presence, empty-input handling
- `test_output_parser.py` — citation extraction + dedup + first-occurrence ordering, confidence parsing + clamping to [0,1], unknown/hallucinated citation ids ignored, no-citation/no-confidence passthrough
- `test_generator.py` — end-to-end wiring with a fake `OllamaClient`, prompt contains assembled context + query, empty-chunks case, `ValueError` on unsupported `citation_style`
- Full suite: 93 passed (`python -m pytest tests/unit -q`)
- Gap: no test exercises a real Ollama server (same pattern as Phases 3-5) — only manual smoke testing via `scripts/generate.py` against the real Qdrant collection covers that.

## Config Keys Used
- `configs/base.yaml` → `generation.model`, `generation.ollama_host`, `generation.temperature`, `generation.citation_style`

## Open Items / Deferred
- `footnote` and `list` citation styles are declared in config but not implemented — `inline` is the only supported value; revisit only if a real use case needs them.
- No retry/fallback if the LLM omits all `[n]` markers or the confidence line — `output_parser` degrades gracefully (empty citations / `None` confidence) rather than raising, which is correct behavior for a genuinely unanswerable question but indistinguishable from a model that just forgot to cite.
- Phase 7 (Evaluation — RAGAS + retrieval metrics) is next; `src/pipeline.py` remains a stub until then.
