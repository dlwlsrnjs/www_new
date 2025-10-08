import argparse
import json
from pathlib import Path

from submission.vendor.rekgmcts.mcts import MCTSPathFinder
from submission.vendor_adapter.tkg_backend import TKGBackend
import submission.vendor_adapter as vad


class DummyLLM:
    def __call__(self, prompt):
        # Always return a moderate score text; replace with real API in production
        return ["0.8\nRelevant to the question."]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parents[2] / "submission" / "sample" / "demo.jsonl"))
    args = parser.parse_args()

    src = Path(args.input)
    total = 0
    answered = 0
    correct = 0

    for rec in load_jsonl(src):
        total += 1
        question = rec.get("question", "")
        backend = TKGBackend(rec.get("prompt", ""), question=question)
        vad.current_backend = backend

        # monkey patch hooks expected by MCTS
        import submission.vendor.rekgmcts.mcts as mcts_mod
        mcts_mod.relation_search_prune = lambda entity_id, entity_name, pre_relations, pre_head, question, llm: backend.relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, llm)
        mcts_mod.entity_search = lambda entity, relation, head=True: backend.entity_search(entity, relation, head)
        mcts_mod.get_entity_name = lambda entity_id: backend.get_entity_name(entity_id)
        mcts_mod.llm = DummyLLM()

        # topic entities: pin to target tail for immediate-after demo
        meta = backend.get_temporal_meta(question)
        topic_entities = {meta["target_tail"]: meta["target_tail"]} if meta else {"E0": "E0"}

        # rule pre-screen: if exact immediate-after exists, return it directly
        pred = None
        path = []
        if meta:
            pivot_end = meta["pivot_end"]
            segs = backend.get_target_segments(meta["target_rel"], meta["target_tail"])
            exact = [(h, s, e) for (h, s, e) in segs if s == pivot_end]
            if exact:
                h, s, e = sorted(exact, key=lambda x: (x[2] - x[1], x[0]))[0]
                pred = h
                path = [(h, meta["target_rel"], meta["target_tail"])]
            else:
                later = [(h, s, e) for (h, s, e) in segs if s > pivot_end]
                ordered = sorted(later, key=lambda x: (x[1], x[2] - x[1], x[0]))
                whitelist = [t[0] for t in ordered[:2]]
                backend.allowed_heads = set(whitelist)

        if pred is None:
            finder = MCTSPathFinder(
                question=question,
                topic_entities=topic_entities,
                llm=mcts_mod.llm,
                max_depth=3,
                num_retain_entity=2,
                max_iterations=5,
                score_threshold=0.8,
                exploration_constant=0.5,
                prune_min_candidates=2,
            )
            path = finder.search()
            pred = path[-1][0] if path else None
        backend.allowed_heads = None

        label = rec.get("label")
        ok = int(pred == label)
        answered += 1 if pred is not None else 0
        correct += ok
        print(json.dumps({"pred": pred, "label": label, "correct": ok, "path": path}, ensure_ascii=False))

    print(json.dumps({
        "total": total,
        "answered": answered,
        "accuracy_on_answered": (correct/answered) if answered else 0.0,
        "overall_accuracy": (correct/total) if total else 0.0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


