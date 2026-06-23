# Phase 4 — Query Processing

## Summary
Turns one raw user question into the inputs Phase 5 (Retrieval) actually searches with: an intent label (used to decide whether decomposition is worth running), a HyDE hypothetical passage + N paraphrases (widen dense/sparse recall), and — for multi-hop questions — a set of independently-answerable sub-questions. No vector store or LLM-generation work happens here; this phase only produces strings that get embedded/searched/decomposed downstream. State left behind: none on disk — pure library code called per-query at request time.

## Worked Example
Query: **"What was Apple's net sales in fiscal 2023?"** (Note: this phase originally ran on local Ollama/`llama3.2:3b`; everything below is the *current* real output after the later Ollama→Groq migration — see CLAUDE.md "LLM Provider Migration Notes" — since `GroqClient` is a drop-in replacement with the same `complete()` interface, nothing about this phase's logic changed, only which LLM answers the prompts.)

1. `QueryAnalyzer(cfg).analyze(query)` sends the classification system prompt + this query to the LLM. Real output:
   ```python
   AnalyzedQuery(
       raw_query="What was Apple's net sales in fiscal 2023?",
       intent="factual_lookup",
       is_multi_hop=False,
   )
   ```
   Correct call: this is a single-fact lookup ("what is X"), not a comparison/calculation/multi-hop question — `is_multi_hop=False` means `QueryDecomposer` won't be invoked at all downstream (see step 3).
2. `QueryTransformer(cfg).transform(query)` makes two more LLM calls. Real output:
   ```python
   TransformedQuery(
       raw_query="What was Apple's net sales in fiscal 2023?",
       hyde_doc=(
           "Our net sales for fiscal 2023 were $383.0 billion, representing a 7.8% "
           "increase from $355.3 billion in fiscal 2022, driven primarily by growth "
           "in our iPhone and services segments, which contributed $191.0 billion "
           "and $68.4 billion to net sales, respectively, for the fiscal year ended "
           "September 30, 2023."
       ),
       multi_queries=[
           "What were Apple's total revenues for the fiscal year 2023?",
           "How much did Apple generate in terms of overall revenue during fiscal 2023?",
           "What was the total amount of money Apple earned from sales in its fiscal year 2023?",
       ],
   )
   ```
   Worth noticing: the HyDE passage is *plausible but wrong* — it invents "$383.0 billion" and a "7.8% increase" that don't match the real filing (actual: $383,285 million, a 3% *decrease*). That's fine and expected — per `query_transformer.py`'s docstring, this passage is "only used to find real matching text, never shown to a user." Its job is to land near the *real* answer in embedding space by using similar financial vocabulary/structure, not to be factually correct itself.
3. `QueryDecomposer(cfg).decompose(query)` is never even called here — Phase 5's orchestration only invokes it when `analyzed.is_multi_hop` is `True` (see `scripts/generate.py`), and step 1 returned `False`. Had this been "Compare Apple's and Microsoft's net sales growth in FY2023" instead, `is_multi_hop` would (ideally) come back `True` and decomposition would split it into independently-retrievable sub-questions like `["What was Apple's net sales growth in FY2023?", "What was Microsoft's net sales growth in FY2023?"]`.

Phase 5 takes the raw query + this `TransformedQuery` (5 distinct strings total: 1 raw + 1 HyDE + 3 multi-query) and searches with all of them — see its Worked Example.

## Files Changed / Added
- `src/utils/llm_client.py` — new — `OllamaClient`: thin wrapper over `ollama.Client.chat`, single prompt in / string out, shared by this phase and (later) Phase 6 generation
- `src/query/models.py` — new — `QueryIntent`, `AnalyzedQuery`, `TransformedQuery`, `DecomposedQuery` pydantic models
- `src/query/query_analyzer.py` — modified (stub → full) — `QueryAnalyzer.analyze()`: zero-shot LLM intent classification into 6 labels
- `src/query/query_transformer.py` — modified (stub → full) — `QueryTransformer`: `generate_hyde()`, `generate_multi_queries()`, `transform()`
- `src/query/query_decomposer.py` — modified (stub → full) — `QueryDecomposer.decompose()`: multi-hop sub-question splitting, gated by config
- `configs/base.yaml` — modified — `generation.model` corrected from `llama3.2` to `llama3.2:3b` (the actual installed Ollama tag — `ollama list` never registers a bare `llama3.2` alias)
- `.env.example` — modified — same `llama3.2:3b` correction in `OLLAMA_MODEL`/`LLM_MODEL` and the `ollama pull` instruction
- `tests/unit/test_llm_client.py` — new — `OllamaClient.complete()` message construction (system+user), options passthrough
- `tests/unit/test_query_analyzer.py` — new — exact label match, label embedded in a sentence, unknown response defaults to `other`
- `tests/unit/test_query_transformer.py` — new — HyDE stripping, multi-query blank/duplicate filtering + cap at N, per-flag skip behavior
- `tests/unit/test_query_decomposer.py` — new — disabled passthrough (no LLM call made), multi-line split, empty-response fallback

## Key Design Decisions
- **Intent classification gates decomposition, not retrieval strategy in general**: HyDE and multi-query help every question, so they're controlled purely by static config flags (`use_hyde`, `use_multi_query`). Decomposition only pays off for genuinely multi-hop questions — running it on a simple lookup just adds an LLM round-trip and can fragment a perfectly good single query into noisy partial ones. `AnalyzedQuery.is_multi_hop` is the signal Phase 5's orchestration will use to decide whether to call `QueryDecomposer` at all.
- **6 intent labels (`factual_lookup`, `comparison`, `calculation`, `multi_hop`, `definition`, `other`) instead of the placeholder stub's 4** (`factual, analytical, comparison, temporal`): the stub docstring was scaffold-only filler, not a spec. These 6 map directly to real FinanceBench-style question shapes and — critically — include `multi_hop` explicitly, since that's the one label anything downstream actually branches on.
- **Substring match for parsing the LLM's intent response, not exact match**: local 3B models (`llama3.2:3b`) don't reliably emit just the bare label despite the system prompt — sometimes wrap it in a sentence ("The label is: comparison."). Matching the first known label found anywhere in the lowercased response is more robust than requiring an exact match, and falls back to `other` rather than raising.
- **`QueryDecomposer.decompose()` returns `[raw_query]` (not an empty list) when decomposition is disabled or the LLM returns nothing parseable**: every caller downstream (Phase 5 retrieval-per-sub-question) can loop over `sub_questions` unconditionally — a single-element list naturally degrades to "retrieve once for the original query" with no separate branch.
- **`OllamaClient` pulled into `src/utils/` rather than duplicated inside `query_transformer.py` and (later) `generation/generator.py`**: both phases need identical "one prompt → one string" behavior with no other behavior (no streaming, no multi-turn history) — one shared wrapper means Phase 6 reuses this instead of re-wiring `ollama.Client`.
- **HyDE and multi-query use different temperatures (0.3 / 0.5) than intent classification (0.0)**: classification needs a single deterministic label; HyDE/multi-query exist specifically to introduce *lexical diversity* around the same underlying question, so some sampling temperature is the point.
- **Multi-query dedup drops exact (case-insensitive) matches against the original query only, not against each other**: the LLM is asked for paraphrases, not for N *distinct from each other* — testing this is an actual scaffold problem.

## Execution Flow
One real run (see Manual Smoke Test below) — programmatic, no CLI script for this phase since it's pure library code consumed by Phase 5, not a standalone pipeline stage:
1. `load_config()` resolves `configs/base.yaml` → `cfg.generation.model = "llama3.2:3b"`, `cfg.generation.ollama_host`, `cfg.query.*`
2. `QueryAnalyzer(cfg)` constructed → builds its own `OllamaClient(model=cfg.generation.model, host=cfg.generation.ollama_host)`
3. `analyzer.analyze(query)` → sends classification system+user prompt → `ollama.Client.chat()` → `_parse_intent()` → `AnalyzedQuery`
4. `QueryTransformer(cfg).transform(query)` → if `cfg.query.use_hyde`: one LLM call for a hypothetical passage; if `cfg.query.use_multi_query`: one LLM call for `cfg.query.num_multi_queries` paraphrases, filtered/capped → `TransformedQuery`
5. `QueryDecomposer(cfg).decompose(query)` → if `cfg.query.use_decomposition` is `False` (the base.yaml default): returns immediately with `sub_questions=[query]`, no LLM call; if `True`: one LLM call → split into lines → `DecomposedQuery`

## Data Contract (Input → Output)
- Input: a raw query string (from a user or, later, `scripts/run_query.py` / an eval harness) — no pydantic model, just `str`
- Output (consumed by Phase 5 Retrieval): `AnalyzedQuery` (intent + multi-hop flag), `TransformedQuery` (`hyde_doc: str | None`, `multi_queries: list[str]`), `DecomposedQuery` (`sub_questions: list[str]`) — Phase 5 will embed/search `raw_query` + `hyde_doc` + each `multi_queries` entry (dense+sparse, RRF-fused) per sub-question in `sub_questions`

## Tests
- `tests/unit/test_llm_client.py` — `OllamaClient.complete()` builds the right `messages` list (system optional) and forwards `temperature` via `options`; `ollama.Client` faked, no real server call
- `tests/unit/test_query_analyzer.py` — exact label, label embedded in a longer sentence, unknown text → `other`, `raw_query` passthrough
- `tests/unit/test_query_transformer.py` — HyDE output stripped of whitespace, multi-query blank-line + self-duplicate filtering, cap at `n`, `use_hyde`/`use_multi_query` independently skippable
- `tests/unit/test_query_decomposer.py` — disabled flag makes zero LLM calls and returns `[raw_query]`, multi-line LLM response split correctly, empty/whitespace-only response falls back to `[raw_query]`
- Full suite: 63 passed (`python -m pytest tests/unit -q`) — 48 from Phases 1-3, +15 new this phase
- Gap: no test exercises a real Ollama server (by design, same as Phase 3's "no test loads the real BGE-M3 model") — covered instead by the manual smoke test below
- Known model-accuracy gap (not a code bug): the installed `llama3.2:3b` sometimes misclassifies a genuinely multi-hop question as `other` instead of `multi_hop` (observed during smoke testing) — `_parse_intent()`'s parsing logic is correct, the underlying 3B model's zero-shot classification just isn't perfectly reliable. Revisit if Phase 5/7 numbers show decomposition under-triggering; options are a larger local model or few-shot examples in `_SYSTEM_PROMPT`.

## Config Keys Used
- `configs/base.yaml` → `generation.model`, `generation.ollama_host`, `query.use_hyde`, `query.use_multi_query`, `query.num_multi_queries`, `query.use_decomposition`

## Open Items / Deferred
- No CLI/script entry point for this phase — by design, there's no standalone "query processing" pipeline stage analogous to `build_index.py`; Phase 5's retrieval orchestration (and the `/retrieve` command) will call `QueryAnalyzer`/`QueryTransformer`/`QueryDecomposer` directly.
- Decomposition is currently gated only by the static `cfg.query.use_decomposition` flag, not by `AnalyzedQuery.is_multi_hop` — wiring "only decompose if the analyzer says multi-hop AND the flag is on" is Phase 5's orchestration job, not this phase's.
- `QueryTransformer.generate_multi_queries()` dedupes paraphrases against the original query only, not against each other — a model could in principle emit the same paraphrase twice; harmless (just one redundant retrieval call) but not explicitly guarded.
- `src/pipeline.py` still a stub — full end-to-end wiring still deferred until Phase 5/6 exist (per Phase 3's note, now one phase closer).
