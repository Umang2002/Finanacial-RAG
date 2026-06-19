# Phase N — <Name>

> Copy this file to `phaseN_<name>.md` when phase finishes. Fill every section. Keep it factual — derive from actual diff/files, not memory.

## Summary
2-4 sentences: what this phase does in the pipeline, why it exists, what state it leaves behind on disk (e.g. `data/processed/...`).

## Files Changed / Added
List every file touched this phase, one line each:
- `path/to/file.py` — new|modified — one-line purpose

## Key Design Decisions
- Decision — why (only non-obvious ones; skip boilerplate choices)

## Execution Flow
Step-by-step trace of one real run, e.g.:
1. `scripts/foo.py --args` invoked
2. loads config from `configs/base.yaml`
3. `ClassX.method()` called → does Y
4. writes output to `data/.../out.json`
5. CLI prints rich summary table

## Data Contract (Input → Output)
- Input: which pydantic model / files, from which previous phase
- Output: which pydantic model / files, consumed by which next phase

## Tests
- `tests/unit/test_x.py` — what's covered
- Known gaps (if any)

## Config Keys Used
- `configs/base.yaml` section.key — purpose

## Open Items / Deferred
Anything explicitly stubbed/deferred and why (e.g. PDF parsing not needed yet).
