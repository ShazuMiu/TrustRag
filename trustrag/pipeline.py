"""End-to-end pipeline + the system presets used for baselines and ablations.

    query
      -> hybrid retrieval (BM25 + dense, RRF)
      -> rerank (cross-encoder)
      -> retrieval gate (CORRECT / AMBIGUOUS / INCORRECT) -> corrective fallback / abstain
      -> context packing (dedup, label S1..Sk)
      -> citation-enforced generation
      -> claim extraction (+ ghost-citation detection)
      -> windowed NLI verification against cited + all sources
      -> correction (re-cite / prune / rewrite / remove)
      -> answer policy (answered / partial / abstained)  -> log
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import deep_merge, resolve_path
from .generation.llm import LLM, build_llm
from .generation.prompts import GENERATION_SYSTEM, NAIVE_SYSTEM, generation_user
from .index.store import CorpusIndex
from .logging_db import RunLogger
from .retrieval.gate import CORRECT, INCORRECT, RetrievalGate, rewrite_query
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import Reranker
from .retrieval.web import web_search
from .schemas import ClaimVerdict, Passage, PipelineResult
from .text import tokenize
from .verification.claims import extract_claims, is_abstention
from .verification.corrector import Corrector, apply_policy
from .verification.nli import NLIModel, build_nli
from .verification.verifier import Verifier

# Overrides applied on top of the loaded config to obtain each compared system.
SYSTEM_PRESETS: Dict[str, dict] = {
    "naive": {  # classic RAG baseline: dense top-k -> LLM, no checks
        "retrieval": {"use_bm25": False, "use_dense": True},
        "reranker": {"enabled": False},
        "retrieval_gate": {"enabled": False},
        "pipeline": {"prompt": "naive", "verify": False},
        "correction": {"enabled": False},
    },
    "trustrag": {},
    # ---- ablations (each removes ONE component from trustrag)
    "no_correction": {"pipeline": {"verify": True}, "correction": {"enabled": False}},
    "no_gate": {"retrieval_gate": {"enabled": False}},
    "no_rerank": {"reranker": {"enabled": False}},
    "dense_only": {"retrieval": {"use_bm25": False}},
    "bm25_only": {"retrieval": {"use_dense": False}},
}

PIPELINE_DEFAULTS = {"prompt": "strict", "verify": True}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / max(1, len(sa | sb))


class TrustRAGPipeline:
    def __init__(self, cfg: dict, index: Optional[CorpusIndex] = None, llm: Optional[LLM] = None,
                 nli: Optional[NLIModel] = None, system: str = "trustrag", log: bool = True):
        self.base_cfg = cfg
        self.system = system
        self.cfg = deep_merge(cfg, SYSTEM_PRESETS.get(system, {}))
        self.cfg["pipeline"] = deep_merge(PIPELINE_DEFAULTS, self.cfg.get("pipeline", {}))
        self.index = index or CorpusIndex.load(resolve_path(cfg["index_dir"]), cfg)
        self.llm = llm or build_llm(cfg)
        nc = cfg["nli"]
        self.nli = nli or build_nli(nc["backend"], nc.get("model", ""), nc.get("device", "cpu"),
                                    nc.get("batch_size", 16), nc.get("max_length", 512))
        self.retriever = HybridRetriever(self.index, self.cfg)
        self.reranker = Reranker(self.index, self.cfg)
        self.gate = RetrievalGate(self.cfg)
        self.verifier = Verifier(self.nli, self.cfg)
        self.corrector = Corrector(self.verifier, self.cfg, self.llm)
        self.logger = None
        if log and cfg.get("logging", {}).get("enabled", True):
            self.logger = RunLogger(resolve_path(cfg["logging"]["sqlite_path"]))

    def with_system(self, system: str) -> "TrustRAGPipeline":
        """Same index / models, different preset (cheap: no reloading)."""
        return TrustRAGPipeline(self.base_cfg, self.index, self.llm, self.nli, system, log=False)

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, timings: Dict[str, float]):
        t = time.perf_counter()
        cands = self.retriever.search(query)
        timings["retrieve"] = time.perf_counter() - t

        t = time.perf_counter()
        ranked = self.reranker.rerank(query, cands)
        timings["rerank"] = time.perf_counter() - t

        actions: List[str] = []
        decision = self.gate.assess(ranked, query) if self.gate.enabled else self.gate.assess(ranked)
        t = time.perf_counter()
        if self.gate.enabled:
            rounds = 0
            while decision != CORRECT and rounds < self.gate.max_rounds:
                rounds += 1
                extra: List[Passage] = []
                if rounds == 1:
                    q2 = rewrite_query(query, ranked, self.llm)
                    actions.append(f"query_rewrite: {q2}")
                    extra = self.retriever.search(q2, n=self.retriever.n * 2)
                elif self.cfg["web_fallback"].get("enabled"):
                    try:
                        extra = web_search(query, self.cfg["web_fallback"].get("provider", "semantic_scholar"),
                                           self.cfg["web_fallback"].get("max_results", 5))
                        actions.append(f"web_search: {len(extra)} results")
                    except Exception as exc:
                        actions.append(f"web_search_failed: {exc}")
                else:
                    break
                seen = {p.chunk.chunk_id for p in ranked}
                pool = ranked + [p for p in extra if p.chunk.chunk_id not in seen]
                ranked = self.reranker.rerank(query, pool)  # always score against the ORIGINAL query
                decision = self.gate.assess(ranked, query)
            missing = self.gate.missing_entities(query, ranked)
            if missing:
                actions.append("missing_entities: " + ", ".join(missing))
            ranked = self.gate.filter(ranked)
        timings["gate"] = time.perf_counter() - t

        # pack context: drop near-duplicates, keep final_k, assign citation labels
        final_k = self.cfg["retrieval"].get("final_k", 5)
        thr = self.cfg["retrieval"].get("dedup_threshold", 0.92)
        packed: List[Passage] = []
        for p in ranked:
            if all(_jaccard(p.chunk.text, q.chunk.text) < thr for q in packed):
                packed.append(p)
            if len(packed) >= final_k:
                break
        for i, p in enumerate(packed, 1):
            p.label = f"S{i}"
        return packed, decision, actions

    # ------------------------------------------------------------ main entry
    def run(self, query: str) -> PipelineResult:
        query = " ".join(str(query).split())
        if not query:
            raise ValueError("Query cannot be empty")
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        passages, decision, actions = self.retrieve(query, timings)
        pc = self.cfg["pipeline"]

        def finish(res: PipelineResult) -> PipelineResult:
            timings["total"] = time.perf_counter() - t0
            res.timings = timings
            if self.logger:
                try:
                    self.logger.log(res)
                except Exception:
                    pass
            return res

        if not passages or (self.gate.enabled and decision == INCORRECT and self.gate.abstain_if_incorrect):
            return finish(PipelineResult(query=query, answer=self.cfg["policy"]["abstain_message"],
                                         status="abstained", passages=passages, retrieval_decision=decision,
                                         retrieval_actions=actions + ["abstain: insufficient evidence"],
                                         system=self.system, stats={"abstain_reason": "retrieval"}))

        # ---- generation
        t = time.perf_counter()
        system_prompt = NAIVE_SYSTEM if pc["prompt"] == "naive" else \
            GENERATION_SYSTEM.format(max_sentences=self.cfg["generation"].get("max_sentences", 6))
        raw = self.llm.complete(system_prompt, generation_user(query, passages))
        timings["generate"] = time.perf_counter() - t

        if is_abstention(raw):
            return finish(PipelineResult(query=query, answer=self.cfg["policy"]["abstain_message"],
                                         status="abstained", original_answer=raw, passages=passages,
                                         retrieval_decision=decision, retrieval_actions=actions,
                                         system=self.system, stats={"abstain_reason": "generator"}))

        labels = [p.label for p in passages]
        claims = extract_claims(raw, labels)

        if not pc["verify"]:
            return finish(PipelineResult(query=query, answer=raw, status="answered", original_answer=raw,
                                         passages=passages, retrieval_decision=decision,
                                         retrieval_actions=actions, confidence=0.0, system=self.system,
                                         stats={"n_claims": len(claims)}))

        # ---- verification + correction
        t = time.perf_counter()
        verdicts: List[ClaimVerdict] = []
        all_scores = []
        for c in claims:
            v, scores = self.verifier.verify(c, passages)
            verdicts.append(v)
            all_scores.append(scores)
        timings["verify"] = time.perf_counter() - t

        t = time.perf_counter()
        for v, scores in zip(verdicts, all_scores):
            self.corrector.correct(query, v, scores, passages)
        timings["correct"] = time.perf_counter() - t

        if not self.corrector.enabled:
            status, answer, conf = "answered", raw, \
                (sum(v.entailment for v in verdicts) / len(verdicts) if verdicts else 0.0)
        else:
            pol = apply_policy(verdicts, self.cfg)
            status, answer, conf = pol["status"], pol["answer"], pol["confidence"]

        stats = {
            "n_claims": len(verdicts),
            "labels": {k: sum(v.label == k for v in verdicts)
                       for k in ("supported", "misattributed", "unsupported", "contradicted")},
            "actions": {k: sum(v.action == k for v in verdicts)
                        for k in ("kept", "pruned_citations", "recited", "rewritten", "removed")},
            "ghost_citations": sum(len(v.claim.ghost_citations) for v in verdicts),
        }
        return finish(PipelineResult(query=query, answer=answer, status=status, original_answer=raw,
                                     passages=passages, verdicts=verdicts, retrieval_decision=decision,
                                     retrieval_actions=actions, confidence=conf, system=self.system,
                                     stats=stats))


def build_pipeline(cfg: dict, system: str = "trustrag") -> TrustRAGPipeline:
    return TrustRAGPipeline(cfg, system=system)
