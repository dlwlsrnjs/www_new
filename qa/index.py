import re
from collections import defaultdict


class PerContextIndex:
    """
    Minimal TKG index for demo purposes.
    Parses prompt_text lines with format:
      E<HEAD> R<REL> E<TAIL> [<START>,<END>]
    Example:
      E57 R11 E76 [1,2]

    Builds:
      - by_rel_tail[(rel, tail)] -> list[(head, start, end)]
      - by_hrt[(head, rel, tail)] -> list[(start, end)]
    """

    def __init__(self, prompt_text: str):
        self.by_rel_tail = defaultdict(list)
        self.by_hrt = defaultdict(list)
        if not prompt_text:
            return
        pattern = re.compile(r"\b(E\d+)\s+(R\d+)\s+(E\d+)\s*\[(\d+),(\d+)\]")
        for line in prompt_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            head, rel, tail, s, e = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
            self.by_rel_tail[(rel, tail)].append((head, s, e))
            self.by_hrt[(head, rel, tail)].append((s, e))
        # sort sequences by start then duration for consistency
        for key, seq in list(self.by_rel_tail.items()):
            self.by_rel_tail[key] = sorted(seq, key=lambda x: (x[1], (x[2]-x[1]), x[0]))
        for key, seq in list(self.by_hrt.items()):
            self.by_hrt[key] = sorted(seq, key=lambda x: (x[0], (x[1]-x[0])))


