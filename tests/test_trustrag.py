"""Test suite (stdlib unittest; also runs under pytest).

    python -m unittest discover -s tests -v
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trustrag.config import load_config  # noqa: E402
from trustrag.evaluation.metrics import (citation_metrics, retrieval_metrics, rouge_l,  # noqa: E402
                                         token_f1)
from trustrag.generation.llm import LLM, ExtractiveLLM  # noqa: E402
from trustrag.index.bm25 import BM25  # noqa: E402
from trustrag.index.store import CorpusIndex  # noqa: E402
from trustrag.ingest.chunker import chunk_document, chunk_documents  # noqa: E402
from trustrag.ingest.loaders import load_path  # noqa: E402
from trustrag.pipeline import TrustRAGPipeline  # noqa: E402
from trustrag.retrieval.gate import RetrievalGate, salient_phrases, salient_terms  # noqa: E402
from trustrag.schemas import Chunk, Document, Passage  # noqa: E402
from trustrag.text import extract_numbers, split_sentences  # noqa: E402
from trustrag.verification.claims import extract_claims, render_claim  # noqa: E402
from trustrag.verification.corrector import Corrector, apply_policy  # noqa: E402
from trustrag.verification.nli import LexicalNLI  # noqa: E402
from trustrag.verification.verifier import Verifier  # noqa: E402

CFG = load_config("config/lite.yaml", {"logging": {"enabled": False}})


def P(label, text, doc="d"):
    return Passage(chunk=Chunk(chunk_id=f"{doc}::{label}", doc_id=doc, title="", text=text), label=label)


PASSAGES = [
    P("S1", "The Padma Bridge has a length of 6.15 kilometres. It was opened to road traffic on 25 June 2022.", "padma"),
    P("S2", "K2 is the second-highest mountain on Earth, with a height of 8,611 metres.", "k2"),
    P("S3", "Photosynthesis takes place in chloroplasts. The oxygen comes from splitting water.", "photo"),
]


class TestText(unittest.TestCase):
    def test_sentences_protect_abbreviations(self):
        s = split_sentences("Gao et al. built ALCE. It uses NLI, e.g. TRUE. Done.")
        self.assertEqual(len(s), 3)

    def test_numbers(self):
        self.assertEqual(extract_numbers("8,848.86 m and 57% in 2020"), ["8848.86", "57", "2020"])


class TestIngest(unittest.TestCase):
    def test_chunker_keeps_sentences_whole_and_overlaps(self):
        doc = Document("d1", "T", " ".join(f"Sentence number {i} is here." for i in range(40)))
        chunks = chunk_document(doc, max_words=30, overlap_sentences=1)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertTrue(c.text.endswith("."))
        self.assertEqual(chunks[0].text.split(". ")[-1].rstrip("."), chunks[1].text.split(". ")[0].rstrip("."))
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))

    def test_loader_front_matter(self):
        docs = load_path(str(ROOT / "data/demo/corpus"))
        padma = [d for d in docs if d.doc_id == "padma_bridge"][0]
        self.assertEqual(padma.year, 2023)
        self.assertNotIn("---", padma.text)


class TestBM25(unittest.TestCase):
    def test_ranking(self):
        bm = BM25().fit(["the cat sat on the mat", "dogs chase cats", "stock market news today"])
        top = bm.search("cat on mat", 3)
        self.assertEqual(top[0][0], 0)
        bm2 = BM25.from_state(bm.state())
        self.assertAlmostEqual(bm2.scores("market")[2], bm.scores("market")[2])


class TestClaims(unittest.TestCase):
    def test_parse_citations_and_ghosts(self):
        ans = "The bridge is 6.15 kilometres long [S1]. K2 is 8,611 metres high [S2][S7]. Oxygen comes from water. [3]"
        cl = extract_claims(ans, ["S1", "S2", "S3"])
        self.assertEqual(len(cl), 3)
        self.assertEqual(cl[0].citations, ["S1"])
        self.assertEqual(cl[1].citations, ["S2"])
        self.assertEqual(cl[1].ghost_citations, ["S7"])
        self.assertEqual(cl[2].citations, ["S3"])  # citation after the full stop is re-attached
        self.assertNotIn("[", cl[0].text)

    def test_abstention_and_render(self):
        self.assertEqual(extract_claims("INSUFFICIENT_EVIDENCE"), [])
        self.assertEqual(render_claim("A fact.", ["S1", "S2"]), "A fact [S1][S2].")


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.v = Verifier(LexicalNLI(), CFG)

    def check(self, text, cites):
        c = extract_claims(text + " " + "".join(f"[{x}]" for x in cites) + ".", ["S1", "S2", "S3"])[0]
        return self.v.verify(c, PASSAGES)

    def test_supported(self):
        self.assertEqual(self.check("The Padma Bridge has a length of 6.15 kilometres", ["S1"])[0].label, "supported")

    def test_misattributed(self):
        v, _ = self.check("The Padma Bridge has a length of 6.15 kilometres", ["S2"])
        self.assertEqual(v.label, "misattributed")
        self.assertEqual(v.best_label, "S1")

    def test_numeric_distortion_is_caught(self):
        v, _ = self.check("The Padma Bridge has a length of 9.15 kilometres", ["S1"])
        self.assertEqual(v.label, "contradicted")

    def test_negation_is_contradiction(self):
        v, _ = self.check("Photosynthesis does not take place in chloroplasts", ["S3"])
        self.assertEqual(v.label, "contradicted")

    def test_unsupported(self):
        self.assertEqual(self.check("Bananas are rich in potassium and grow in tropical climates", ["S1"])[0].label,
                         "unsupported")


class FakeLLM(LLM):
    is_generative = True
    name = "fake"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, temperature=None, max_tokens=None):
        return self.reply


class TestCorrector(unittest.TestCase):
    def run_one(self, text, cites, llm=None):
        v = Verifier(LexicalNLI(), CFG)
        corr = Corrector(v, CFG, llm)
        c = extract_claims(text + " " + "".join(f"[{x}]" for x in cites) + ".", ["S1", "S2", "S3"])[0]
        verdict, scores = v.verify(c, PASSAGES)
        return corr.correct("q", verdict, scores, PASSAGES)

    def test_recite(self):
        r = self.run_one("K2 is the second-highest mountain on Earth", ["S1"])
        self.assertEqual(r.action, "recited")
        self.assertEqual(r.final_citations, ["S2"])

    def test_remove_hallucination_without_llm(self):
        self.assertEqual(self.run_one("Bananas are rich in potassium and grow in tropical climates", ["S1"]).action,
                         "removed")

    def test_rewrite_with_llm(self):
        llm = FakeLLM("The Padma Bridge has a length of 6.15 kilometres [S1].")
        r = self.run_one("The Padma Bridge has a length of 9.15 kilometres", ["S1"], llm)
        self.assertEqual(r.action, "rewritten")
        self.assertIn("6.15", r.final_text)

    def test_policy_abstains_when_everything_removed(self):
        r = self.run_one("Bananas are rich in potassium and grow in tropical climates", ["S1"])
        self.assertEqual(apply_policy([r], CFG)["status"], "abstained")


class TestGate(unittest.TestCase):
    def test_salient(self):
        self.assertIn("Jamuna", salient_terms("How long is the Jamuna Bridge?"))
        self.assertEqual(salient_phrases("How long is the Jamuna Bridge in Bangladesh?"), [["Jamuna", "Bridge"]])

    def test_missing_entity_forces_incorrect(self):
        g = RetrievalGate(CFG)
        p = P("S1", "The Padma Multipurpose Bridge is 6.15 kilometres long.")
        p.score = 0.9
        self.assertEqual(g.assess([p], "How long is the Padma Bridge?"), "correct")
        self.assertEqual(g.assess([p], "How long is the Jamuna Bridge?"), "incorrect")


class TestMetrics(unittest.TestCase):
    def test_retrieval(self):
        m = retrieval_metrics(["a", "b", "c"], ["b"], ks=(1, 3))
        self.assertEqual(m["recall@1"], 0.0)
        self.assertEqual(m["recall@3"], 1.0)
        self.assertAlmostEqual(m["mrr"], 0.5)

    def test_text_overlap(self):
        self.assertAlmostEqual(token_f1("the cat sat", "cat sat"), 1.0)
        self.assertGreater(rouge_l("x b c d", "x c d"), 0.8)

    def test_citation_metrics(self):
        judge = Verifier(LexicalNLI(), CFG)
        good = citation_metrics("The Padma Bridge has a length of 6.15 kilometres [S1].", PASSAGES, judge)
        self.assertEqual(good["citation_recall"], 1.0)
        self.assertEqual(good["citation_precision"], 1.0)
        bad = citation_metrics("The Padma Bridge has a length of 6.15 kilometres [S1][S2].", PASSAGES, judge)
        self.assertEqual(bad["citation_precision"], 0.5)
        ghost = citation_metrics("The Padma Bridge has a length of 6.15 kilometres [S9].", PASSAGES, judge)
        self.assertEqual(ghost["ghost_citation_rate"], 1.0)


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        docs = load_path(str(ROOT / "data/demo/corpus"))
        idx = CorpusIndex.build(chunk_documents(docs), CFG, cls.tmp)
        cls.pipe = TrustRAGPipeline(CFG, index=idx, log=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_index_roundtrip(self):
        idx = CorpusIndex.load(self.tmp, CFG)
        self.assertEqual(len(idx), len(self.pipe.index))

    def test_answer_with_verified_citations(self):
        r = self.pipe.run("How long is the Padma Bridge?")
        self.assertIn(r.status, ("answered", "partial"))
        self.assertIn("6.15", r.answer)
        self.assertTrue(all(v.label == "supported" for v in r.verdicts if v.action != "removed"))
        self.assertEqual(r.passages[0].chunk.doc_id, "padma_bridge")

    def test_abstain_on_unanswerable(self):
        self.assertEqual(self.pipe.run("How many parameters does GPT-3 have?").status, "abstained")

    def test_naive_system_does_not_verify(self):
        r = self.pipe.with_system("naive").run("How long is the Padma Bridge?")
        self.assertEqual(r.verdicts, [])

    def test_extractive_llm_prompt_roundtrip(self):
        from trustrag.generation.prompts import generation_user

        out = ExtractiveLLM().complete("", generation_user("How long is the Padma Bridge?", PASSAGES))
        self.assertIn("[S1]", out)

    def test_flask_api(self):
        from app.server import create_app

        cfg = dict(CFG)
        cfg["index_dir"] = str(self.tmp)
        client = create_app(cfg).test_client()
        self.assertEqual(client.get("/api/health").status_code, 200)
        r = client.post("/api/ask", json={"question": "Who first climbed K2?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("claims", r.get_json())
        self.assertEqual(client.post("/api/ask", json={}).status_code, 400)


class TestConfig(unittest.TestCase):
    def test_env_override(self):
        os.environ["TRUSTRAG__LLM__MODEL"] = "llama3.1:8b"
        try:
            self.assertEqual(load_config("config/lite.yaml")["llm"]["model"], "llama3.1:8b")
        finally:
            del os.environ["TRUSTRAG__LLM__MODEL"]


if __name__ == "__main__":
    unittest.main()


class TestGenerativePathWithMockServer(unittest.TestCase):
    """Exercises the OpenAI-compatible client + claim rewriting using scripts/mock_llm_server.py."""

    @classmethod
    def setUpClass(cls):
        import threading

        from werkzeug.serving import make_server

        sys.path.insert(0, str(ROOT / "scripts"))
        import mock_llm_server

        cls.server = make_server("127.0.0.1", 0, mock_llm_server.app)
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        os.environ["MOCK_KEY"] = "x"
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        cls.cfg = load_config("config/lite.yaml", {"logging": {"enabled": False}, "llm": {
            "provider": "openai_compatible", "base_url": f"http://127.0.0.1:{cls.port}/v1", "api_key_env": "MOCK_KEY",
            "max_retries": 0}})
        cls.tmp = Path(tempfile.mkdtemp())
        idx = CorpusIndex.build(chunk_documents(load_path(str(ROOT / "data/demo/corpus"))), cls.cfg, cls.tmp)
        cls.pipe = TrustRAGPipeline(cls.cfg, index=idx, log=False)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_errors_from_unreliable_llm_are_fixed(self):
        r = self.pipe.run("How long is the Padma Bridge?")
        self.assertNotIn("99 percent", r.answer)                  # invented sentence removed
        self.assertTrue(all(v.action != "kept" or v.label == "supported" for v in r.verdicts))
        judge = Verifier(LexicalNLI(), self.cfg)
        m = citation_metrics(r.answer, r.passages, judge)
        self.assertEqual(m["citation_precision"], 1.0)
        naive = self.pipe.with_system("naive").run("How long is the Padma Bridge?")
        self.assertIn("99 percent", naive.answer)
