"""Reciprocal-rank fusion of the parallel retrieval paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from octop.infra.knowledge.index import Hit

# Standard RRF constant (kb-test initial value; retune on octop corpora).
RRF_K = 60


def dedupe_key(hit: Hit) -> tuple[str, str]:
    """Identity across engines: doc plus its strongest anchor, chunk id last.

    Windows split from one anchored unit share a key, so a section that only
    ranks weakly in many windows cannot outvote a strong single hit.
    """
    if hit.page is not None:
        return (hit.doc_id, f"p{hit.page}")
    if hit.anchor:
        return (hit.doc_id, hit.anchor)
    return (hit.doc_id, hit.chunk_id)


def rrf_fuse(named_lists: Mapping[str, Sequence[Hit]], *, k: int = RRF_K) -> list[Hit]:
    """Fuse ranked lists by each entry's best rank across all engines.

    Every key contributes only its best position (never a sum of votes), and
    ties resolve by engine insertion order — ``vector`` first when passed in
    that order — then by list position.
    """
    best: dict[tuple[str, str], tuple[float, int, Hit]] = {}
    for engine_order, hits in enumerate(named_lists.values()):
        for rank, hit in enumerate(hits):
            key = dedupe_key(hit)
            score = 1.0 / (k + rank + 1)
            current = best.get(key)
            if current is None or score > current[0]:
                best[key] = (score, engine_order, hit)
    ranked = sorted(best.values(), key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked]
