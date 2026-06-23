# Phase 2 — Processing

## Summary
Turns each `ParsedFiling` (Phase 1 output) into a list of `Chunk` objects ready for embedding. Cleans residual Unicode/boilerplate noise, serializes raw HTML tables into markdown + natural-language sentences, splits section/table text via a configurable strategy (fixed/recursive/semantic/sentence_window/parent_child), and stamps filing+section identity onto every chunk. Output lands at `data/processed/{ticker}/{form}_{year}/chunks.json` — this is what Phase 3 (Indexing) embeds into Qdrant.

## Worked Example
Continuing the AAPL FY2023 10-K "Fiscal Year Highlights" passage from Phase 1's `parsed.json`. Recall the cleaned section text from Phase 1:
```
...An additional week is included in the first fiscal quarter every five or six
years to realign the Company's fiscal quarters with calendar quarters, which
occurred in the first quarter of 2023. The Company's fiscal year 2023 spanned
53 weeks, whereas fiscal years 2022 and 2021 spanned 52 weeks each. Fiscal Year
Highlights The Company's total net sales were $383.3 billion and net income
was $97.0 billion during 2023. The Company's total net sales decreased 3% or
$11.0 billion during 2023 compared to 2022...
```
1. `TextCleaner.clean()` runs first — for this passage it's a no-op (no stray Unicode quotes/dashes or ToC cross-references here), but it does fire elsewhere in the same `Item 7` section, e.g. normalizing the curly apostrophe in `Company's` (`’` → `'`).
2. `chunk_text(cleaned, cfg)` dispatches to `chunk_recursive()` (the configured default strategy) with `chunk_size=512`, `chunk_overlap=64`. `RecursiveCharacterTextSplitter` walks the configured separators (`\n\n`, `\n`, `. `, ` `) and cuts this section into multiple ~512-char windows with 64-char overlap. Our exact passage lands inside real chunk `chunk_index=2` of the `Item 7` section — here's the **actual chunk text** produced (`chunk_id=33c5dee5-4d03-4b10-aa04-1f28a2b8ecea`, 85 tokens by the word-count proxy):
   ```
   . An additional week is included in the first fiscal quarter every five or six
   years to realign the Company's fiscal quarters with calendar quarters, which
   occurred in the first quarter of 2023. The Company's fiscal year 2023 spanned
   53 weeks, whereas fiscal years 2022 and 2021 spanned 52 weeks each. Fiscal Year
   Highlights The Company's total net sales were $383.3 billion and net income
   was $97.0 billion during 2023. The Company's total net sales decreased 3% or
   $11.0 billion during 2023 compared to 2022
   ```
   Note the leading `. ` — that's the tail of the *previous* chunk's last sentence, an artifact of the 64-char overlap window, not a cleaning bug.
3. The same dollar figure also exists in tabular form elsewhere in the filing (e.g. a "Total net sales | $383,285" row in the segment-breakdown table) — that table goes through `TableSerializer` instead, producing a *separate* chunk like `"Total net sales (2023): $. Total net sales (): 383,285..."` (NL-sentence form) plus a markdown-table chunk. Both the prose chunk above and these table chunks end up indexed in Phase 3 — this is why Phase 5's retrieval later returns multiple chunks (prose Item 7, Item 8 financial statements, and Table 19) all answering the same net-sales question.
4. `enrich()` stamps `ChunkMetadata` onto the chunk: `ticker="AAPL"`, `cik="0000320193"`, `form="10-K"`, `fiscal_year=2023`, `company_name="Apple Inc."`, `item_label="Item 7"`, `section_title="Management's Discussion and Analysis of Financial Condition and..."`.
5. Written to `data/processed/AAPL/10-K_2023/chunks.json` as one of 737 total chunks for this filing — this exact chunk (and its table-chunk siblings) is what Phase 3 embeds next.

## Files Changed / Added
- `src/processing/models.py` — new — `Chunk`, `ChunkMetadata`, `ChunkStrategy` literal
- `src/processing/cleaner.py` — modified (stub → full) — `TextCleaner`: Unicode normalization, stray "Table of Contents" removal, whitespace collapse
- `src/processing/table_serializer.py` — modified (stub → full) — `TableSerializer`: raw HTML `<table>` → markdown + row-wise NL sentences
- `src/processing/chunkers.py` — modified (stub → full) — `chunk_fixed`, `chunk_recursive`, `chunk_sentence_window`, `chunk_parent_child`, `chunk_semantic`, `chunk_text()` dispatcher
- `src/processing/enricher.py` — modified (stub → full) — `enrich()`: stamps `ChunkMetadata` onto chunks
- `scripts/process_filings.py` — new — CLI: `parsed.json` → clean → chunk → enrich → `chunks.json`
- `configs/base.yaml` — modified — added `chunking.sentence_window_size`, `chunking.semantic_similarity_threshold`, `chunking.parent_chunk_size`, `chunking.child_chunk_size`
- `tests/unit/test_chunkers.py` — new (filled stub) — all 5 strategies + dispatcher
- `tests/unit/test_cleaner.py` — new
- `tests/unit/test_table_serializer.py` — new
- `tests/unit/test_enricher.py` — new
- `notebooks/02_chunking_experiments.ipynb` — modified (stub → full, executed) — runs `TextCleaner`, `TableSerializer`, and all chunking strategies against real AAPL 10-K 2023 data; compares chunk count/size per strategy; justifies `recursive` default

## Key Design Decisions
- **Cleaner is narrow in scope**: `html_parser.py` (Phase 1) already collapses all whitespace to single spaces and strips the ToC block via its gap heuristic. Cleaner only handles what survives that: Unicode quote/dash normalization, stray inline "Table of Contents" cross-references. Deliberately does NOT strip standalone page-number digits — once text is flattened to single-line, a digit run is indistinguishable from a real dollar figure; a false-positive strip would silently corrupt financial data.
- **Raw tables become synthetic sections, not a separate code path**: `ParsedFiling.raw_tables` is filing-level, not tied to an Item section. `process_filings.py` wraps each serialized table as a `ParsedSection(item_label="Table N", ...)` so it flows through the exact same `chunk_text()` + `enrich()` pipeline as prose — no parallel table-chunking logic.
- **Table serializer keeps empty cells, only drops fully-blank rows**: first attempt dropped empty `<td>` cells as "padding," which shifted later cells left and broke the header→value column alignment used to build NL sentences (caught by `test_table_serializer.py` failing during implementation). Fixed by keeping cell positions intact; a blank corner cell above the row-label column is structurally meaningful, not padding.
- **Table serializer emits both markdown and NL sentences**: markdown preserves row/column structure for the LLM at generation time; NL sentences ("Net sales (2023): $383,285") exist because BGE-M3 was never trained to treat pipe-delimited syntax as semantically equivalent to prose — dense retrieval against natural-language financial questions matches sentences far better.
- **Sentence splitting via regex, not nltk/spacy**: `_SENTENCE_SPLIT_RE` (punctuation + whitespace + capital/quote/paren lookahead) is good enough for 10-K/10-Q prose and avoids a new dependency just for `sentence_window`/`semantic` strategies.
- **`chunk_text()` token_count is a word-count proxy, not a real tokenizer**: no tokenizer dependency wired in yet — Phase 3's embedding model does real tokenization at encode time. This field is only for chunk-size sanity logging.
- **Semantic chunker lazy-loads BGE-M3 via a module-level singleton** (`_get_embedder`): avoids the ~2.3GB download/init cost for every other strategy. Tests monkeypatch `_get_embedder` with a fake returning hand-picked vectors — never touches the real model, keeps tests offline/fast.
- **Parent-child indexes children, fetches parents at generation time**: small chunks (`child_chunk_size`) embed precisely for retrieval match; large chunks (`parent_chunk_size`) carry context but dilute embeddings. `is_parent=True` chunks are not meant to be vector-indexed in Phase 3 — looked up by `parent_id` instead.
- **Sentence_window splits `text` (embedded) from `context_text` (shown to LLM)**: embedding a single sentence gets a tight match; `context_text` (±N sentences) is what generation actually reads. Keeping them separate fields prevents diluting embedding precision.
- **`chunk_text(text, cfg)` takes the whole DictConfig, not unpacked args**: each strategy needs a different subset of `chunking.*` keys — centralizing the unpacking means `process_filings.py` doesn't need to know which keys matter for which strategy.
- **`enrich()` is a separate step from chunking, mutates in place**: keeps `chunkers.py` a pure text-in/chunks-out module, testable without constructing a `ParsedFiling`. `enrich()` returns the same list object (mutated), matching the cheap in-place pattern used throughout this phase.

## Execution Flow
1. `python scripts/process_filings.py --ticker AAPL --years 2023` (or no args → config defaults)
2. `load_config()` resolves `configs/base.yaml`
3. `_find_parsed_filings()` scans `data/processed/{ticker}/{form}_{year}/` for `parsed.json` (written by Phase 1's `parse_filings.py`); skips if `chunks.json` already exists (idempotent)
4. per filing, `chunk_filing(filing, cfg, cleaner, serializer)`:
   a. for each `ParsedSection` in `filing.sections`: `TextCleaner.clean(section.text)` → if non-empty, `chunk_text(cleaned, cfg)` dispatches to the configured strategy → `enrich(chunks, filing, section)` stamps `ChunkMetadata`
   b. `TableSerializer.serialize_all(filing.raw_tables)` → each serialized table wrapped as a synthetic `ParsedSection(item_label="Table N")` → same `chunk_text()` + `enrich()` path
   c. all chunks from (a) and (b) concatenated into one list
5. writes `data/processed/{ticker}/{form}_{year}/chunks.json` (`[c.model_dump(mode="json") for c in chunks]`)
6. CLI prints rich summary table (ticker/form/year/strategy/chunk count/status)

### Strategy dispatch detail (`chunk_text`)
- `fixed` → `chunk_fixed()`: char windows of `chunk_size` with `chunk_overlap` step
- `recursive` → `chunk_recursive()`: wraps `langchain_text_splitters.RecursiveCharacterTextSplitter` with config `separators`
- `sentence_window` → `chunk_sentence_window()`: one chunk per sentence, `context_text` = ±`sentence_window_size` neighbors
- `parent_child` → `chunk_parent_child()`: `RecursiveCharacterTextSplitter` at `parent_chunk_size`, then each parent re-split at `child_chunk_size`; children carry `parent_id`
- `semantic` → `chunk_semantic()`: sentences embedded via lazy-loaded BGE-M3, consecutive sentences grouped while cosine similarity ≥ `semantic_similarity_threshold` and group length < `chunk_size`

## Data Contract (Input → Output)
- Input (from Phase 1): `data/processed/{ticker}/{form}_{year}/parsed.json` — `ParsedFiling` model
- Output (consumed by Phase 3): `data/processed/{ticker}/{form}_{year}/chunks.json` — `list[Chunk]`, each with `text`, `strategy`, `chunk_index`, `token_count`, `parent_id`/`is_parent` (parent_child only), `context_text` (sentence_window only), `metadata: ChunkMetadata` (ticker/cik/form/fiscal_year/company_name/item_label/section_title)

## Tests
- `tests/unit/test_chunkers.py` — fixed (size/overlap, empty input), recursive (paragraph/sentence splitting), sentence_window (context window clamping at boundaries), parent_child (parent_id linkage), semantic (similarity-based grouping + single-sentence short-circuit, both via fake embedder), `chunk_text()` dispatch + unknown-strategy error
- `tests/unit/test_cleaner.py` — empty input, Unicode normalization, ToC removal, whitespace collapse, dollar-figure preservation
- `tests/unit/test_table_serializer.py` — markdown output, NL sentence output, empty/single-row table handling, `serialize_all` dropping unparseable tables
- `tests/unit/test_enricher.py` — metadata stamped correctly onto every chunk, in-place mutation (same list object returned)
- Full suite: 35 passed (`python -m pytest tests/unit -q`)
- Gap: no test runs the real BGE-M3 model (by design — fake embedder substitutes it); no test exercises `process_filings.py`'s file-finding/caching logic against real `parsed.json` fixtures on disk

## Config Keys Used
- `configs/base.yaml` → `chunking.strategy`, `chunking.chunk_size`, `chunking.chunk_overlap`, `chunking.separators`, `chunking.sentence_window_size`, `chunking.semantic_similarity_threshold`, `chunking.parent_chunk_size`, `chunking.child_chunk_size`, `embeddings.dense_model` (semantic strategy only)

## Open Items / Deferred
- `src/pipeline.py` left as a stub — full end-to-end wiring deferred until Phase 3-6 (indexing/retrieval/generation) exist; wiring it now against unbuilt phases would be speculative
- Token counting is a word-count proxy, not a real tokenizer — acceptable since Phase 3's embedder does real tokenization; revisit only if exact token budgets become necessary
- Table serializer's NL-sentence heuristic assumes label-in-column-0 + period-headers-in-row-0 — holds for every EDGAR table structure seen so far, not guaranteed by spec
