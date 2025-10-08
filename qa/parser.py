from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class ImmediateAfterQuery:
    pivot_head: str
    pivot_relation: str
    pivot_tail: str
    relation: str
    tail: str


def parse_immediately_after(question: str) -> Optional[ImmediateAfterQuery]:
    """Parse simplified 'immediately after' questions of the form:
    Find the entity that was the Rxx of Eyy immediately after Eaa Rbb Ecc
    where target is (head=?, relation=Rxx, tail=Eyy), pivot is (Eaa Rbb Ecc)
    """
    if not question:
        return None
    # Example pattern; adjust as needed for demo samples
    # Find the entity that was the R11 of E76 immediately after E57 R11 E76
    m = re.search(r"Find the entity that was the\s+(R\d+)\s+of\s+(E\d+)\s+immediately after\s+(E\d+)\s+(R\d+)\s+(E\d+)", question)
    if not m:
        return None
    target_rel, target_tail, pivot_head, pivot_rel, pivot_tail = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    return ImmediateAfterQuery(
        pivot_head=pivot_head,
        pivot_relation=pivot_rel,
        pivot_tail=pivot_tail,
        relation=target_rel,
        tail=target_tail,
    )


