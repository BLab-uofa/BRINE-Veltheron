from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agent.schemas import ObserverResponse
from config import load_config
from literature.bond_client import BondClient
from literature.evidence import (
    build_evidence_packet,
    compare_conductivity_values,
    summarize_evidence,
)
from observer.analytics import campaign_summary, recent_trials, top_trials
from observer.flags import generate_flags
from observer.loader import load_brine_csv
from shadow.decision_log import load_decision_log
from shadow.status import live_pipeline_status
from shadow.view import build_shadow_view

load_dotenv()


SYSTEM_RULES = """
You are the read-only scientific interface for the BRINE self-driving electrolyte laboratory.
Your purpose is to help a scientist understand experimental results, causal CHARIOT shadow decisions,
trends, and relevant literature. You do NOT control the laboratory.

CONTROL BOUNDARY
- Never start, stop, repeat, choose, or alter a BRINE experiment.
- Never modify BO, the digital twin, CHARIOT thresholds, robot settings, potentiostat settings, files, or code.
- CHARIOT is SHADOW-ONLY. Its decisions did not control BRINE.
- The digital twin predicts CONDUCTIVITY ONLY. It does not predict HER, OER, stability window, or LSV outcome.
- CHARIOT uses a frozen pre-trial conductivity prediction plus measured EIS to judge whether conductivity evidence
  is sufficient for RUN_LSV / REVIEW / SKIP_LSV. It never claims to predict LSV success.
- REVIEW means decision insufficiency and counts as performing LSV in benchmark comparisons.

BRINE VARIABLES
x1 = NaCl stock amount (uL)
x2 = KCl stock amount (uL)
x3 = ZnCl2 stock amount (uL)
x4 = ZnSO4 stock amount (uL)
y1 = ionic conductivity (S/m)
HER and OER are potentials in V. OER-HER may be reported as a stability-window descriptor.
Total formulation volume is configured as 1200 uL.
Never invent stock concentrations.

NUMERICAL RULES
- Python-calculated statistics and CHARIOT log values are authoritative.
- Do not invent missing measurements or uncertainty values.
- The twin's raw uncertainty is NOT assumed to be a Gaussian sigma.
- normalized_disagreement_score is NOT a z-score. Describe it as a normalized model-measurement disagreement score.
- Old CHARIOT decisions are causal/frozen and must never be reinterpreted as if future trials were known.

CHARIOT INTERPRETATION
- A SKIP means the measured conductivity is below target and the calibrated twin evidence also supports a confident
  below-target decision.
- A REVIEW means automatic triage is not sufficiently supported, for example cold start, prediction band crossing
  the target, or strong twin/EIS disagreement.
- A RUN means conductivity qualification is met with acceptable model/measurement consistency.
- A potentially avoidable LSV is counterfactual: BRINE collected LSV data although shadow CHARIOT would have skipped.
  It does not prove the original experiment was scientifically wrong.
- Do not present CHARIOT as a raw-savings winner over the blind fixed threshold. The fixed threshold necessarily skips
  every below-target EIS value; CHARIOT intentionally sacrifices some of that maximum savings to escalate uncertain cases.

LITERATURE
Valency Bond literature is external evidence. Distinguish BRINE measurement, literature observation, quantitative
benchmark, and scientific hypothesis. Respect paper similarity classifications exactly. Never invent a literature value.
If no directly comparable numerical benchmark exists, say: "No directly comparable numerical literature value was found in the retrieved papers."
Prefer direct/very-close/close analogues and avoid citing low-similarity papers when better evidence exists.
Mechanisms in analogue papers are hypotheses for BRINE unless the campaign directly establishes them.
Do not request literature on your own. Literature lookup is only performed when the scientist explicitly asks for
literature, papers, published work, reported values, Valency Bond, or a similar external-literature comparison.

MEASUREMENT INSUFFICIENCY
Composition + EIS + LSV do not necessarily identify a unique physical mechanism. If the available measurements cannot
separate mechanisms, state measurement insufficiency. Do not convert correlation into causation.

CONVERSATION
Recent dialogue may be provided only to resolve conversational references such as "it", "that", "those", or
"compare it with literature". Never treat numerical values from prior dialogue as authoritative if the freshly rebuilt
Python campaign context provides newer values. Fresh campaign context always wins.

LIVE CAMPAIGN STATUS
- If live_pipeline_status is present, use it as the authoritative current handoff state.
- A trial may be waiting for the twin prediction, waiting for EIS, or ready but not yet frozen.
- Never describe a pending trial as decided.
- The scientist may ask questions while BRINE is still running; always answer from the freshly reloaded current state.

PLOTS
Do not request or generate a plot unless the user explicitly asks for a plot, graph, chart, figure, or visualization.
"""


class GeminiObserver:
    def __init__(
        self,
        config_path: str | Path | None = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        bond_client: Optional[BondClient] = None,
    ):
        self.cfg = load_config(config_path)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY in your .env file.")

        from google import genai
        from google.genai import types

        timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))
        retry_attempts = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "1"))
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=retry_attempts),
            ),
        )
        self.bond_client = bond_client or BondClient()

        # Lightweight conversational state only. Fresh campaign data are
        # still rebuilt from disk before every answer.
        self.current_topic: str | None = None
        self.dialogue_history: list[dict[str, str]] = []
        self.max_history_turns = 6

        # Exposed read-only status for the dashboard. This lets the UI show
        # whether a literature lookup actually happened, instead of making
        # the scientist infer that from the prose answer.
        self.last_literature_status: dict = {
            "requested": False,
            "queried": False,
            "papers_retrieved": 0,
            "papers_used": 0,
            "query": None,
        }

    def _context(
        self,
        campaign_file: Path,
        decision_log_file: Path,
        prediction_log_file: Path | None = None,
    ) -> dict:
        df = load_brine_csv(campaign_file, self.cfg)
        decisions = load_decision_log(decision_log_file)
        shadow_view = build_shadow_view(df, decisions)
        flags = [f.to_dict() for f in generate_flags(df, shadow_view, self.cfg)]
        context = {
            "summary": campaign_summary(df, shadow_view, self.cfg),
            "recent_trials": recent_trials(
                df, shadow_view, min(int(self.cfg["observer"].get("recent_rows", 10)), 5)
            ),
            "top_3_conductivity": top_trials(df, "conductivity", 3, True),
            "flags": flags[:15],
            "decision_log": str(decision_log_file),
            "read_only": True,
            "shadow_only": True,
            "column_mapping": self.cfg["columns"],
            "stock_configuration": self.cfg["stocks"],
        }
        if prediction_log_file is not None:
            context["live_pipeline_status"] = live_pipeline_status(
                campaign_file, prediction_log_file, decision_log_file, self.cfg
            )
            context["prediction_log"] = str(prediction_log_file)
        return context

    def _explicit_literature_request(self, question: str) -> bool:
        """
        Return True only when the user explicitly asks for literature.

        Important: use word-boundary matching for short tokens such as
        DOI so words like "doing" do not accidentally trigger Bond.
        """

        q = question.lower().strip()

        phrases = [
            "literature",
            "paper",
            "papers",
            "published",
            "publication",
            "publications",
            "reported value",
            "reported values",
            "reported in",
            "compare with literature",
            "compare to literature",
            "compare it with literature",
            "compare it to literature",
            "compare with papers",
            "compare to papers",
            "valency",
            "bond",
            "has this been reported",
            "has it been reported",
            "has anyone reported",
            "find studies",
            "find study",
            "find research",
            "research papers",
        ]

        if any(phrase in q for phrase in phrases):
            return True

        # Match DOI as a standalone token only.
        # This prevents "doing" from matching "doi".
        if re.search(r"\bdoi\b", q):
            return True

        return False

    def _infer_topic(self, question: str) -> str | None:
        q = question.lower().strip()

        # Explicit topic words always win.
        if any(
            term in q
            for term in [
                "conductivity",
                "ionic conductivity",
                "eis",
                "s/m",
            ]
        ):
            return "conductivity"

        if any(term in q for term in ["chariot", "shadow", "run_lsv", "review_lsv", "skip_lsv"]):
            return "chariot"

        if any(term in q for term in ["her", "hydrogen evolution"]):
            return "her"

        if any(term in q for term in ["oer", "oxygen evolution"]):
            return "oer"

        if any(term in q for term in ["lsv", "stability window", "electrochemical stability"]):
            return "stability"

        if any(
            term in q
            for term in [
                "composition",
                "formulation",
                "x1",
                "x2",
                "x3",
                "x4",
                "nacl",
                "kcl",
                "zncl2",
                "znso4",
            ]
        ):
            return "composition"

        # In this campaign, an unqualified request for the current max/min/
        # mean refers to the main screening property: ionic conductivity.
        if any(
            phrase in q
            for phrase in [
                "current maximum",
                "current max",
                "maximum value",
                "max value",
                "current minimum",
                "current min",
                "minimum value",
                "mean value",
                "current mean",
                "current average",
            ]
        ):
            return "conductivity"

        # Pronoun-only follow-ups preserve the last known topic.
        referential_phrases = [
            "compare it",
            "compare this",
            "compare that",
            "why is it",
            "why was it",
            "plot it",
            "graph it",
            "show it",
            "what about it",
            "tell me more",
        ]
        if any(phrase in q for phrase in referential_phrases):
            return self.current_topic

        return None

    def _update_topic(self, question: str) -> str | None:
        inferred = self._infer_topic(question)
        if inferred is not None:
            self.current_topic = inferred
        return self.current_topic

    def _build_bond_query(self, question: str, topic: str | None = None) -> str:
        q = question.lower()
        resolved_topic = topic or self.current_topic

        chemistry = (
            "aqueous electrolyte NaCl KCl ZnCl2 ZnSO4 zinc chloride zinc sulfate mixed salt electrolyte "
        )

        if (
            resolved_topic == "conductivity"
            or "conductivity" in q
            or "ionic conductivity" in q
        ):
            return (
                chemistry
                + "ionic conductivity concentration dependence ion transport viscosity ion pairing"
            )

        if resolved_topic == "her" or "her" in q or "hydrogen evolution" in q:
            return chemistry + "hydrogen evolution HER zinc aqueous electrolyte electrochemical stability"

        if resolved_topic == "oer" or "oer" in q or "oxygen evolution" in q:
            return chemistry + "oxygen evolution OER aqueous electrolyte electrochemical stability"

        if (
            resolved_topic == "stability"
            or "stability" in q
            or "window" in q
            or "lsv" in q
        ):
            return (
                chemistry
                + "electrochemical stability window hydrogen evolution oxygen evolution aqueous zinc electrolyte"
            )

        if resolved_topic == "chariot":
            return (
                "self-driving laboratory autonomous laboratory characterization decision layer "
                "measurement triage expensive destructive characterization uncertainty experiment selection"
            )

        if resolved_topic == "composition":
            return chemistry + "mixed salt composition electrolyte properties aqueous zinc"

        return chemistry + "aqueous zinc electrolyte electrochemical properties transport properties"

    def _prioritize_evidence(self, evidence_packet: list[dict]) -> list[dict]:
        strong = {"direct_match", "very_close_analogue", "close_analogue"}
        strong_papers = [p for p in evidence_packet if p.get("similarity_level") in strong]
        if strong_papers:
            return strong_papers
        broad = [p for p in evidence_packet if p.get("similarity_level") == "broad_zinc_analogue"]
        if broad:
            return broad
        return evidence_packet[:3]

    def _get_literature_evidence(self, query: str, context: dict, limit: int = 6) -> dict:
        if not self.bond_client.configured:
            raise RuntimeError("Bond was requested, but BOND_BEARER_TOKEN is not configured in .env.")

        print("\n[Valency Bond] Searching literature...")
        print(f"[Valency Bond] Search query: {query}")
        t0 = time.perf_counter()
        raw = self.bond_client.search(query, limit=limit)
        elapsed = time.perf_counter() - t0

        full_packet = build_evidence_packet(raw)
        summary = summarize_evidence(full_packet)
        packet = self._prioritize_evidence(full_packet)

        campaign_max = (
            context.get("summary", {})
            .get("outputs", {})
            .get("ionic_conductivity_S_per_m", {})
            .get("max")
        )
        comparisons = []
        if campaign_max is not None:
            comparisons = compare_conductivity_values(float(campaign_max), packet)

        print(f"[Valency Bond] {len(full_packet)} papers returned in {elapsed:.2f} s")
        counts = summary["similarity_counts"]
        print(
            "[Evidence] "
            f"{counts['direct_match']} direct, {counts['very_close_analogue']} very close, "
            f"{counts['close_analogue']} close, {counts['broad_zinc_analogue']} broad Zn, "
            f"{counts['low_similarity']} low similarity"
        )
        print(
            f"[Evidence] {summary['papers_with_explicit_numeric_conductivity']} papers contain explicit numerical conductivity values"
        )
        print(f"[Evidence] {len(packet)} papers passed the relevance filter for Gemini")

        return {
            "bond_query": query,
            "bond_search_confidence": raw.get("confidence"),
            "bond_warnings": raw.get("warnings", []),
            "evidence_summary": summary,
            "papers": packet,
            "conductivity_comparisons": comparisons,
            "evidence_rules": {
                "numeric_values_must_be_explicit": True,
                "python_handles_unit_conversion": True,
                "analogues_are_not_direct_matches": True,
                "direct_benchmark_requires_comparable_chemistry": True,
                "low_similarity_filtered_when_better_evidence_exists": True,
            },
        }

    def _paper_ids_used_in_answer(self, answer: str) -> list[str]:
        """
        Return paper IDs actually referenced by Gemini, preserving
        first-use order.

        Example:
            "The ZnCl2 study [P2] ... [P1]" -> ["P2", "P1"]
        """
        found = re.findall(r"\[(P\d+)\]", answer or "")

        seen: set[str] = set()
        ordered: list[str] = []

        for paper_id in found:
            if paper_id not in seen:
                seen.add(paper_id)
                ordered.append(paper_id)

        return ordered

    def _safe_http_url(self, value: object) -> str | None:
        """Return only normal http/https URLs for dashboard links."""
        if value is None:
            return None

        url = str(value).strip()
        if re.match(r"^https?://", url, flags=re.IGNORECASE):
            return url

        return None

    def _doi_parts(self, value: object) -> tuple[str | None, str | None]:
        """Return a clean DOI label and deterministic doi.org URL."""
        if value is None:
            return None, None

        doi = str(value).strip()
        if not doi:
            return None, None

        doi = re.sub(
            r"^https?://(?:dx\.)?doi\.org/",
            "",
            doi,
            flags=re.IGNORECASE,
        )
        doi = re.sub(
            r"^doi:\s*",
            "",
            doi,
            flags=re.IGNORECASE,
        ).strip()

        if not doi:
            return None, None

        return doi, f"https://doi.org/{doi}"

    def _append_literature_sources(
        self,
        answer: str,
        literature_evidence: dict | None,
    ) -> str:
        """
        Append clickable literature sources using only validated Bond
        evidence. Gemini may cite [P1], [P2], etc., but Python creates
        the DOI and source links deterministically.
        """
        if not literature_evidence:
            return answer

        papers = literature_evidence.get("papers", []) or []
        if not papers:
            return answer

        paper_lookup = {
            str(paper.get("paper_id")): paper
            for paper in papers
            if paper.get("paper_id")
        }

        used_ids = self._paper_ids_used_in_answer(answer)

        # If Gemini discussed literature but forgot an explicit [P#]
        # marker, show the strongest filtered evidence instead of
        # leaving the answer without an evidence trail.
        if not used_ids:
            used_ids = [
                str(paper.get("paper_id"))
                for paper in papers[:3]
                if paper.get("paper_id")
            ]

        source_blocks: list[str] = []

        for paper_id in used_ids:
            paper = paper_lookup.get(paper_id)
            if not paper:
                continue

            title = str(paper.get("title") or "Untitled paper").strip()
            similarity = str(paper.get("similarity_level") or "").strip()
            source_name = str(paper.get("source") or "").strip()
            doi_label, doi_url = self._doi_parts(paper.get("doi"))
            paper_url = self._safe_http_url(paper.get("url"))

            lines = [f"**[{paper_id}]** *{title}*"]

            metadata = []
            if similarity:
                metadata.append(similarity.replace("_", " "))
            if source_name:
                metadata.append(source_name)
            if metadata:
                lines.append("  \n" + " · ".join(metadata))

            links = []
            if doi_label and doi_url:
                links.append(f"[DOI: {doi_label}]({doi_url})")
            if paper_url:
                links.append(f"[Open source record]({paper_url})")

            if links:
                lines.append("  \n" + " · ".join(links))

            source_blocks.append("\n".join(lines))

        if not source_blocks:
            return answer

        return (
            answer.rstrip()
            + "\n\n---\n\n"
            + "### Sources\n\n"
            + "\n\n".join(source_blocks)
        )

    def _recent_dialogue_for_prompt(self) -> list[dict[str, str]]:
        if not self.dialogue_history:
            return []
        return self.dialogue_history[-self.max_history_turns :]

    def _record_turn(self, question: str, answer: str) -> None:
        self.dialogue_history.append(
            {
                "user": question,
                "assistant": answer,
            }
        )
        if len(self.dialogue_history) > self.max_history_turns:
            self.dialogue_history = self.dialogue_history[-self.max_history_turns :]

    def _ask(
        self,
        question: str,
        context: dict,
        literature_evidence: dict | None = None,
    ) -> ObserverResponse:
        from google.genai import types

        prompt = (
            SYSTEM_RULES
            + "\n\nCURRENT CONVERSATIONAL TOPIC:\n"
            + json.dumps(self.current_topic, default=str)
            + "\n\nRECENT DIALOGUE FOR REFERENT RESOLUTION ONLY:\n"
            + json.dumps(self._recent_dialogue_for_prompt(), indent=2, default=str)
            + "\n\nDETERMINISTIC BRINE + CHARIOT CONTEXT:\n"
            + json.dumps(context, indent=2, default=str)
        )

        if literature_evidence is not None:
            prompt += (
                "\n\nVALIDATED VALENCY BOND EVIDENCE:\n"
                + json.dumps(literature_evidence, indent=2, default=str)
            )
            prompt += """

MANDATORY LITERATURE INTERPRETATION
The literature evidence has already been retrieved, classified, filtered, and normalized by Python.
Do not invent values, modify similarity classes, or expose internal dictionary names.
If no directly comparable numerical benchmark exists, state that naturally.
Focus on the most chemically relevant papers. Cite every literature-derived scientific statement with the validated
paper ID in square brackets, for example [P1]. Include the paper title and DOI in the prose when it materially helps,
but never invent either. When discussing scope for improvement, use the current best measured value, whether the best
point lies near a sampled boundary when supported, whether literature provides a true ceiling, and what remains unknown.
Meeting a screening target does not mean the property is optimized.
If no directly comparable numerical benchmark exists but relevant analogue papers are present, DO NOT stop after saying
that no direct benchmark was found. Continue with the strongest analogue evidence, cite it with [P#], explain what the
analogue does and does not establish for BRINE, and answer the user's improvement question using campaign evidence plus
that literature context.
"""

        prompt += "\n\nUSER QUESTION:\n" + question

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ObserverResponse,
                temperature=0.1,
                max_output_tokens=1400,
            ),
        )
        return ObserverResponse.model_validate_json(response.text)

    def analyze(
        self,
        question: str,
        campaign_file: str | Path,
        decision_log_file: str | Path,
        use_bond: bool = False,
        prediction_log_file: str | Path | None = None,
    ) -> ObserverResponse:
        campaign_file = Path(campaign_file)
        decision_log_file = Path(decision_log_file)
        prediction_path = Path(prediction_log_file) if prediction_log_file is not None else None

        # Fresh state is rebuilt from disk on EVERY turn so an ongoing campaign is never stale.
        context = self._context(campaign_file, decision_log_file, prediction_path)

        # Update the lightweight conversational topic before routing tools.
        topic = self._update_topic(question)

        # IMPORTANT: Bond is explicit-only. Gemini cannot autonomously escalate
        # an ordinary local campaign/CHARIOT question into a literature search.
        literature_requested = self._explicit_literature_request(question)
        self.last_literature_status = {
            "requested": literature_requested,
            "queried": False,
            "papers_retrieved": 0,
            "papers_used": 0,
            "query": None,
        }

        # Do not let Gemini produce a literature-sounding answer when no
        # literature retrieval actually happened. This removes the ambiguous
        # failure mode where the stock "no directly comparable value" sentence
        # could appear even though Bond was disabled.
        if literature_requested and not use_bond:
            raise RuntimeError(
                "Literature comparison requested, but Valency Bond is disabled. "
                "Enable Bond in the dashboard before asking for literature."
            )

        if literature_requested:
            bond_query = self._build_bond_query(question, topic=topic)
            evidence = self._get_literature_evidence(bond_query, context)

            evidence_summary = evidence.get("evidence_summary", {}) or {}
            papers = evidence.get("papers", []) or []
            self.last_literature_status = {
                "requested": True,
                "queried": True,
                "papers_retrieved": int(evidence_summary.get("number_of_papers", 0) or 0),
                "papers_used": len(papers),
                "query": bond_query,
            }

            print("[Gemini] Comparing validated literature with campaign data...")
            result = self._ask(question, context, evidence)
            result.answer = self._append_literature_sources(
                result.answer,
                evidence,
            )
            self._record_turn(question, result.answer)
            return result

        result = self._ask(question, context)
        self._record_turn(question, result.answer)
        return result
