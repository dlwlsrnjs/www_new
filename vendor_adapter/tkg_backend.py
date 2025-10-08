from typing import Dict, List, Optional

from submission.qa.index import PerContextIndex
from submission.qa.parser import parse_immediately_after, ImmediateAfterQuery


class TKGBackend:
    def __init__(self, prompt_text: str, question: Optional[str] = None):
        self.index = PerContextIndex(prompt_text)
        self._rel_cache: Dict = {}
        self._ent_cache: Dict = {}
        self._meta_cache = None
        self._segments_cache: Dict = {}
        self.allowed_heads: Optional[set[str]] = None
        self.q: Optional[ImmediateAfterQuery] = parse_immediately_after(question) if question else None

    def relation_search_prune(self, entity_id: str, entity_name: str, pre_relations: List[str], pre_head: int, question: str, llm=None):
        ck = (entity_id,)
        if ck in self._rel_cache:
            return self._rel_cache[ck]
        target_rel = self.q.relation if self.q else None
        rels = set()
        for (rel, tail), seq in self.index.by_rel_tail.items():
            if target_rel and rel != target_rel:
                continue
            for (head, s, e) in seq:
                if head == entity_id:
                    rels.add((rel, True))
        for (rel, tail), seq in self.index.by_rel_tail.items():
            if target_rel and rel != target_rel:
                continue
            if tail != entity_id:
                continue
            rels.add((rel, False))
        result = [{"entity": entity_id, "relation": r, "score": 1.0, "head": h} for (r, h) in sorted(rels)]
        self._rel_cache[ck] = result
        return result

    def entity_search(self, entity: str, relation: str, head: bool = True) -> List[str]:
        ck = (entity, relation, bool(head))
        if ck in self._ent_cache:
            return self._ent_cache[ck]
        result: List[str] = []
        if head:
            for (rel, tail), seq in self.index.by_rel_tail.items():
                if rel != relation:
                    continue
                for (h, s, e) in seq:
                    if h == entity:
                        if self.q and tail != self.q.tail:
                            continue
                        result.append(tail)
            ret = sorted(set(result))
            self._ent_cache[ck] = ret
            return ret
        else:
            if self.q and relation == self.q.relation:
                segs = []
                for (rel, tail), seq in self.index.by_rel_tail.items():
                    if rel != relation or tail != entity:
                        continue
                    for (h, s, e) in seq:
                        segs.append((h, s, e))
                if not segs:
                    return []
                meta = self.get_temporal_meta()
                pivot_end = meta["pivot_end"] if meta else None
                if pivot_end is not None:
                    head_to_best = {}
                    head_to_exact = {}
                    for (h, s, e) in segs:
                        if s < pivot_end:
                            continue
                        if self.allowed_heads is not None and h not in self.allowed_heads:
                            continue
                        best = head_to_best.get(h)
                        if best is None or s < best[0] or (s == best[0] and (e - s) < (best[1] - best[0])):
                            head_to_best[h] = (s, e)
                        if s == pivot_end:
                            dur = e - s
                            prev = head_to_exact.get(h)
                            if prev is None or dur < prev:
                                head_to_exact[h] = dur
                    if not head_to_best:
                        self._ent_cache[ck] = []
                        return []
                    exact_heads = sorted(head_to_exact.items(), key=lambda x: (x[1], x[0]))
                    exact_order = [h for (h, _) in exact_heads]
                    later_heads = [(h, se[0], se[1]) for (h, se) in head_to_best.items() if h not in head_to_exact]
                    later_heads = sorted(later_heads, key=lambda x: (x[1], (x[2]-x[1]), x[0]))
                    ordered = exact_order + [h for (h, _, _) in later_heads]
                    ret = ordered[:20]
                    self._ent_cache[ck] = ret
                    return ret
        for (rel, tail), seq in self.index.by_rel_tail.items():
            if rel != relation or tail != entity:
                continue
            for (h, s, e) in seq:
                result.append(h)
        ret = sorted(set(result))
        self._ent_cache[ck] = ret
        return ret

    def get_entity_name(self, entity_id: str) -> str:
        return entity_id

    def get_triple_segments(self, head: str, relation: str, tail: str):
        segs = []
        for (rel, t), seq in self.index.by_rel_tail.items():
            if rel != relation or t != tail:
                continue
            for (h, s, e) in seq:
                if h == head:
                    segs.append((s, e))
        return sorted(segs)

    def get_temporal_meta(self, question: Optional[str] = None):
        if self._meta_cache is not None and question is None:
            return self._meta_cache
        qstr = question if question is not None else None
        qobj = parse_immediately_after(qstr) if qstr else self.q
        if not qobj:
            return None
        pivots = self.index.by_hrt.get((qobj.pivot_head, qobj.pivot_relation, qobj.pivot_tail), [])
        if not pivots:
            return None
        pivot_end = max(e for (s, e) in pivots)
        pivot_ends = sorted([e for (s, e) in pivots])
        meta = {
            "pivot_end": pivot_end,
            "pivot_ends": pivot_ends,
            "target_rel": qobj.relation,
            "target_tail": qobj.tail,
        }
        if question is None:
            self._meta_cache = meta
        return meta

    def get_target_segments(self, relation: str, tail: str):
        ck = (relation, tail)
        if ck in self._segments_cache:
            return self._segments_cache[ck]
        segs = []
        for (rel, t), seq in self.index.by_rel_tail.items():
            if rel != relation or t != tail:
                continue
            for (h, s, e) in seq:
                segs.append((h, s, e))
        ret = sorted(segs, key=lambda x: (x[1], (x[2]-x[1]), x[0]))
        self._segments_cache[ck] = ret
        return ret


