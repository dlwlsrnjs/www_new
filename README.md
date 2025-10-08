# Temporal QA via Rule‑Guided MCTS over Temporal KGs

This folder contains a minimal, submission‑ready demo that reproduces the paper’s approach: hybrid rule‑guided search plus MCTS over a Temporal Knowledge Graph (TKG) to answer “immediately after” questions. The demo ships with a Dummy LLM for fully offline runs and can be swapped to GPT‑4 easily.

## Key ideas
- **Rule‑guided pre‑filtering for immediate‑after**: given the pivot triple’s end time `pivot_end`, prioritize segments `(head, target_rel, target_tail)` that start exactly at `pivot_end`. If found, return it (hard override).
- **Rule Top‑k → MCTS re‑ranking**: if no exact match exists, whitelist the earliest `(start > pivot_end)` candidates (Top‑2) and let MCTS re‑rank them via LLM pruning/evaluation.
- **Temporal masking before expansion**: actions leading to segments with `start < pivot_end` are blocked. We also expose `(S,E)` in prompts to enforce strict temporal scoring.
- **Prompt/parse stabilization**: evaluation prompt enforces a numeric‑only first line (`0.0–1.0`) and a one‑sentence rationale; entity pruning returns top‑k names as a comma‑separated list.
- **Caching**: memoization for relation/entity search, temporal meta/segments, and LLM calls (evaluate/prune).

## What’s inside
- `submission/sample/demo.jsonl`: one sample (context triples + question/label)
- `submission/scripts/run_demo.py`: end‑to‑end demo (rules + MCTS pipeline)
- `submission/qa/index.py`: lightweight TKG index (`by_rel_tail`, `by_hrt`)
- `submission/qa/parser.py`: immediate‑after query parser (`ImmediateAfterQuery`)
- `submission/vendor_adapter/tkg_backend.py`: adapter layer with temporal masking/whitelisting
- `submission/vendor/rekgmcts/mcts.py`: MCTS (selection/expansion/eval/prune/UCT) with strict output parsing
- `submission/vendor/rekgmcts/prompts.py`: evaluation/pruning prompt templates
- `submission/vendor/rekgmcts/utils.py`: small parsers (entity name extraction)
- `submission/requirements.txt`: minimal dependencies

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r submission\requirements.txt
```

## Run
```powershell
python submission\scripts\run_demo.py --input submission\sample\demo.jsonl
```
Expected output (for the included sample):
```json
{"pred": "E6", "label": "E6", "correct": 1, "path": [["E6", "R11", "E76"]]}
{
  "total": 1,
  "answered": 1,
  "accuracy_on_answered": 1.0,
  "overall_accuracy": 1.0
}
```

## Method at a glance (aligned with the paper)
1) **TKG indexing**: parse lines `E<HEAD> R<REL> E<TAIL> [<START>,<END>]` into `by_rel_tail[(rel, tail)] → [(head,s,e)]` and `by_hrt[(head,rel,tail)] → [(s,e)]`.
2) **Immediate‑after meta**: for pivot `(pivot_head, pivot_relation, pivot_tail)`, set `pivot_end = max(end)`.
3) **Rule search**:
   - Exact: if any `(h, target_rel, target_tail)` has `start == pivot_end`, return the shortest duration.
   - Non‑exact: sort `(start > pivot_end)` by `(start, duration)` and whitelist Top‑2 `allowed_heads`.
4) **MCTS**:
   - Root is fixed to `target_tail` (topic entity).
   - Only `relation == target_rel` is considered; `start < pivot_end` candidates are masked.
   - If candidates ≥ `prune_min_candidates`, call LLM pruning (`entity_p_prompt`) to select top entities.
   - Children are scored via `EVALUATE_STATE_PROMPT` (STRICT OUTPUT). `(S,E)` may be shown to improve temporal consistency.
   - Standard UCT for selection/backprop.
5) **Disagreement resolver**: if MCTS predicts out‑of‑whitelist or fails, fall back to rule Top‑1.

## Prompts (summary)
- Evaluation (`EVALUATE_STATE_PROMPT`): first line strictly numeric `0.0–1.0`, second line a one‑sentence rationale; may include `(S,E)`.
- Pruning (`entity_p_prompt`): output top‑k entity names separated by commas.

## Swap in GPT‑4 (optional)
The demo uses a **Dummy LLM** to stay offline. To use GPT‑4 instead:
1) Replace `DummyLLM` in `submission/scripts/run_demo.py` with your GPT‑4 client.
2) Set `OPENAI_API_KEY`.
3) Keep the STRICT OUTPUT contract for robust parsing.

## Metrics
- For each record we print `{pred, label, correct, path}` and a short summary at the end.

## Reproduce / extend
- Add more records to `submission/sample/demo.jsonl`.
- For ToT‑scale runs, serialize your TKG into the same text format and reuse `qa/index.py` and the adapter.
- Tune MCTS via `MCTSPathFinder(...)` arguments (depth/iterations/exploration/top‑k).

## Limitations
- This is a minimal demo specialized for immediate‑after queries. Other temporal types (co‑start, strictly‑after, before, etc.) require minor extensions to rules/prompts.
- With Dummy LLM, pruning/evaluation is deterministic and exploration diversity is limited vs. real LLMs.

## License
Distributed for research reproduction. Please comply with external model/data licenses where applicable.
