"""Orchestration for the multi-agent research system.

The pipeline owns the run: it creates the agents, hands each one the context it needs,
records every step in a shared :class:`ResearchState`, and enforces the safety rails
(one revision maximum, real source urls only, no invented citations).

Workflow::

    topic -> SearchAgent -> ReaderAgent -> WriterAgent -> CriticAgent
                                                  ^            |
                                                  |__ revision (max 1)

The state object is deliberately plain data (dataclasses + dicts) so the Streamlit layer
can render it and tests can assert on it without touching the model.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from agents import (
    AgentError,
    BaseAgent,
    ConfigurationError,
    CriticOutput,
    ReaderOutput,
    WriterOutput,
    build_agents,
    find_citations,
    format_excerpts,
    format_notes,
    format_source_index,
    render_sources_section,
    sanitize_report_urls,
    strip_unknown_citations,
)
from tools import (
    MAX_SEARCH_RESULTS,
    PageContent,
    ResearchToolError,
    SearchResult,
)

logger = logging.getLogger(__name__)

#: Hard cap. The Critic may ask for a revision, but never more than this many.
MAX_REVISIONS = 1


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_int(name: str, default: int) -> int:
    """Read an integer setting from the environment, falling back when it is unusable."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s instead.", name, raw, default)
        return default


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------
@dataclass
class AgentStep:
    """One agent execution: what it was given, what it produced, what went wrong."""

    name: str
    display_name: str
    status: str = StepStatus.PENDING.value
    input_summary: str = ""
    summary: str = ""
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_s: float = 0.0
    started_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status,
            "input_summary": self.input_summary,
            "summary": self.summary,
            "output": self.output,
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "started_at": self.started_at,
        }


@dataclass
class ResearchState:
    """Everything the run produced, in a form the UI and the tests can both read."""

    topic: str = ""
    status: str = RunStatus.RUNNING.value
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    steps: List[AgentStep] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    pages: List[Dict[str, Any]] = field(default_factory=list)

    notes: Optional[Dict[str, Any]] = None
    critic_feedback: Optional[Dict[str, Any]] = None
    draft_report: str = ""
    revised_report: Optional[str] = None
    final_report: str = ""
    citations: List[str] = field(default_factory=list)
    revision_count: int = 0

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # -- helpers ------------------------------------------------------------------------
    def add_step(self, step: AgentStep) -> AgentStep:
        self.steps.append(step)
        return step

    def step(self, name: str) -> Optional[AgentStep]:
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def record_error(self, message: str) -> None:
        logger.error(message)
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    @property
    def source_ids(self) -> List[str]:
        return [str(source.get("source_id", "")) for source in self.sources if source.get("source_id")]

    @property
    def usable_pages(self) -> List[Dict[str, Any]]:
        return [page for page in self.pages if page.get("ok")]

    @property
    def succeeded(self) -> bool:
        return self.status in (RunStatus.COMPLETED.value, RunStatus.COMPLETED_WITH_WARNINGS.value)

    def summary_lines(self) -> List[str]:
        """One short line per agent, for the 'what each agent did' section of the UI."""
        lines = []
        for step in self.steps:
            marker = {
                StepStatus.COMPLETED.value: "OK",
                StepStatus.FAILED.value: "FAILED",
                StepStatus.SKIPPED.value: "SKIPPED",
            }.get(step.status, step.status.upper())
            detail = step.summary or step.error or ""
            lines.append(f"{step.display_name} [{marker}] {detail}".strip())
        return lines

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot (contains no credentials)."""
        return {
            "topic": self.topic,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "revision_count": self.revision_count,
            "steps": [step.to_dict() for step in self.steps],
            "sources": list(self.sources),
            "pages": list(self.pages),
            "notes": self.notes,
            "critic_feedback": self.critic_feedback,
            "draft_report": self.draft_report,
            "revised_report": self.revised_report,
            "final_report": self.final_report,
            "citations": list(self.citations),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
class ResearchPipeline:
    """Runs the four agents in order and returns an inspectable state object."""

    def __init__(
        self,
        *,
        agents: Optional[Dict[str, BaseAgent]] = None,
        llm: Any = None,
        search_fn: Optional[Callable[..., List[SearchResult]]] = None,
        fetch_fn: Optional[Callable[..., List[PageContent]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_sources: int = MAX_SEARCH_RESULTS,
        max_pages: int = 4,
        max_revisions: int = MAX_REVISIONS,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.max_sources = max(1, int(max_sources))
        self.max_pages = max(1, int(max_pages))
        # The critic can never authorise more than one revision, whatever the caller asks for.
        self.max_revisions = max(0, min(int(max_revisions), MAX_REVISIONS))
        self._llm = llm
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn
        self._model = model
        self._temperature = temperature
        self._agents = agents
        self._progress_callback = progress_callback

    # -- construction helpers -----------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        *,
        agents: Optional[Dict[str, BaseAgent]] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        **overrides: Any,
    ) -> "ResearchPipeline":
        """Build a pipeline from the environment; explicit keyword arguments win."""
        settings: Dict[str, Any] = {
            "max_sources": _env_int("MAX_SOURCES", MAX_SEARCH_RESULTS),
            "max_pages": _env_int("MAX_PAGES", 4),
        }
        settings.update(overrides)
        return cls(agents=agents, progress_callback=progress_callback, **settings)

    def _ensure_agents(self) -> Dict[str, BaseAgent]:
        if self._agents is None:
            self._agents = build_agents(
                llm=self._llm,
                search_fn=self._search_fn,
                fetch_fn=self._fetch_fn,
                max_sources=self.max_sources,
                max_pages=self.max_pages,
                model=self._model,
                temperature=self._temperature,
            )
        return self._agents

    def _emit(self, stage: str, status: str, message: str) -> None:
        """Report progress to the caller (the Streamlit UI) without ever raising."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(stage, status, message)
        except Exception:  # a broken UI callback must not break a run
            logger.debug("progress callback failed", exc_info=True)

    # -- main entry point ---------------------------------------------------------------
    def run(self, topic: str) -> ResearchState:
        """Execute the full research workflow for ``topic``."""
        topic = (topic or "").strip()
        state = ResearchState(topic=topic, started_at=_now())
        self._emit("pipeline", StepStatus.RUNNING.value, "Starting research run")

        if not topic:
            state.status = RunStatus.FAILED.value
            state.record_error("Please enter a research topic before starting.")
            state.finished_at = _now()
            self._emit("pipeline", StepStatus.FAILED.value, "Empty research topic")
            return state

        try:
            agents = self._ensure_agents()
        except ConfigurationError as exc:
            state.status = RunStatus.FAILED.value
            state.record_error(str(exc))
            state.finished_at = _now()
            self._emit("pipeline", StepStatus.FAILED.value, str(exc))
            return state

        # 1 + 2. Search, then read the retrieved pages -------------------------------
        sources = self._stage_search(state, agents["search"], topic)
        if sources is None:
            return self._finish(state, failed=True)

        pages = self._stage_reader(state, agents["reader"], topic, sources)
        if pages is None:
            return self._finish(state, failed=True)

        # 3. Draft the report -------------------------------------------------------
        reader_output: Optional[ReaderOutput] = (
            ReaderOutput.model_validate(state.notes) if state.notes else None
        )
        evidence_block = self._evidence_block(reader_output, pages)
        source_index = format_source_index(sources)

        draft = self._stage_writer(state, agents["writer"], topic, source_index, evidence_block)
        if draft is None:
            return self._finish(state, failed=True)

        # 4. Review, and revise at most once ---------------------------------------
        critic = self._stage_critic(state, agents["critic"], topic, source_index, draft.report_markdown)
        if critic is not None and critic.significant_issues and self.max_revisions > 0:
            self._stage_revision(
                state,
                agents["writer"],
                topic=topic,
                source_index=source_index,
                evidence_block=evidence_block,
                draft=draft,
                critic=critic,
            )
        elif critic is not None and critic.significant_issues:
            step = state.step("critic")
            if step is not None:
                step.summary += " | revision budget exhausted, report shipped with logged issues"

        # 5. Assemble the final, URL-complete report --------------------------------
        self._finalise_report(state, sources)
        return self._finish(state)

    # -- stages -------------------------------------------------------------------------
    def _stage_search(
        self, state: ResearchState, agent: BaseAgent, topic: str
    ) -> Optional[List[SearchResult]]:
        step = state.add_step(
            AgentStep(
                name="search",
                display_name=agent.display_name,
                status=StepStatus.RUNNING.value,
                input_summary=f"research topic: {topic!r}",
                started_at=_now(),
            )
        )
        self._emit("search", StepStatus.RUNNING.value, "Searching the web for sources")
        started = time.perf_counter()
        try:
            sources: List[SearchResult] = agent.run(topic)  # type: ignore[attr-defined]
        except (ResearchToolError, AgentError) as exc:
            step.status = StepStatus.FAILED.value
            step.error = str(exc)
            step.duration_s = time.perf_counter() - started
            state.record_error(f"Search failed: {exc}")
            self._emit("search", StepStatus.FAILED.value, str(exc))
            return None
        except Exception as exc:  # unexpected, but the UI must still get a clear message
            step.status = StepStatus.FAILED.value
            step.error = f"Unexpected error: {exc.__class__.__name__}: {exc}"
            step.duration_s = time.perf_counter() - started
            state.record_error(f"Search failed unexpectedly: {exc}")
            self._emit("search", StepStatus.FAILED.value, str(exc))
            return None

        step.status = StepStatus.COMPLETED.value
        step.duration_s = time.perf_counter() - started
        step.summary = f"found {len(sources)} usable source(s)"
        step.output = {"sources": [source.to_dict() for source in sources]}
        state.sources = [source.to_dict() for source in sources]
        self._emit("search", StepStatus.COMPLETED.value, step.summary)
        return sources

    def _stage_reader(
        self,
        state: ResearchState,
        agent: BaseAgent,
        topic: str,
        sources: Sequence[SearchResult],
    ) -> Optional[List[PageContent]]:
        step = state.add_step(
            AgentStep(
                name="reader",
                display_name=agent.display_name,
                status=StepStatus.RUNNING.value,
                input_summary=f"{len(sources)} source url(s), fetching up to {self.max_pages}",
                started_at=_now(),
            )
        )
        self._emit("reader", StepStatus.RUNNING.value, "Retrieving and reading the sources")
        started = time.perf_counter()

        try:
            pages: List[PageContent] = agent.retrieve(sources)  # type: ignore[attr-defined]
        except Exception as exc:
            pages = []
            state.add_warning(f"Page retrieval raised {exc.__class__.__name__}: {exc}")

        usable = [page for page in pages if page.ok]
        state.pages = [page.to_dict() for page in pages]
        failed_urls = [page.url for page in pages if not page.ok]
        if failed_urls:
            state.add_warning(
                f"{len(failed_urls)} source(s) could not be retrieved and were skipped: "
                + ", ".join(failed_urls)
            )

        if not usable:
            step.status = StepStatus.FAILED.value
            step.duration_s = time.perf_counter() - started
            step.error = "No source page could be retrieved."
            state.record_error(
                "No usable sources: every retrieved page failed to load, so the report "
                "cannot be grounded in evidence."
            )
            self._emit("reader", StepStatus.FAILED.value, step.error)
            return None

        try:
            notes: ReaderOutput = agent.run(topic, pages)  # type: ignore[attr-defined]
        except (AgentError, ConfigurationError) as exc:
            # Degrade gracefully: the Writer still gets the raw excerpts as evidence.
            step.status = StepStatus.FAILED.value
            step.duration_s = time.perf_counter() - started
            step.error = f"Note extraction failed: {exc}"
            step.summary = f"read {len(usable)} page(s); structured notes unavailable"
            state.notes = None
            state.add_warning(
                "The Reader agent could not produce structured notes; the report was "
                f"written directly from page excerpts ({exc})."
            )
            self._emit("reader", StepStatus.FAILED.value, step.summary)
            return pages

        step.status = StepStatus.COMPLETED.value
        step.duration_s = time.perf_counter() - started
        step.summary = (
            f"read {len(usable)}/{len(pages)} page(s), "
            f"extracted {len(notes.notes)} note set(s), {len(notes.gaps)} gap(s)"
        )
        step.output = notes.model_dump()
        state.notes = notes.model_dump()
        self._emit("reader", StepStatus.COMPLETED.value, step.summary)
        return pages

    def _evidence_block(
        self, reader_output: Optional[ReaderOutput], pages: Sequence[PageContent]
    ) -> str:
        """Evidence handed to the Writer: reader notes when available, else excerpts."""
        parts: List[str] = []
        if reader_output is not None:
            parts.append(format_notes(reader_output))
        parts.append("Raw excerpts from the retrieved pages:")
        parts.append(format_excerpts(pages))
        return "\n\n".join(parts)

    def _stage_writer(
        self,
        state: ResearchState,
        agent: BaseAgent,
        topic: str,
        source_index: str,
        evidence_block: str,
    ) -> Optional[WriterOutput]:
        step = state.add_step(
            AgentStep(
                name="writer",
                display_name=agent.display_name,
                status=StepStatus.RUNNING.value,
                input_summary=f"{len(state.sources)} source(s), notes={'yes' if state.notes else 'no'}",
                started_at=_now(),
            )
        )
        self._emit("writer", StepStatus.RUNNING.value, "Writing the draft report")
        started = time.perf_counter()
        try:
            draft: WriterOutput = agent.run(  # type: ignore[attr-defined]
                topic=topic, source_index=source_index, evidence_block=evidence_block
            )
        except (AgentError, ConfigurationError) as exc:
            step.status = StepStatus.FAILED.value
            step.error = f"Drafting failed: {exc}"
            step.duration_s = time.perf_counter() - started
            state.record_error(f"The Writer agent could not produce a report: {exc}")
            self._emit("writer", StepStatus.FAILED.value, str(exc))
            return None

        step.status = StepStatus.COMPLETED.value
        step.duration_s = time.perf_counter() - started
        step.summary = f"drafted '{draft.title or 'untitled'}' ({len(draft.report_markdown)} chars)"
        step.output = draft.model_dump()
        state.draft_report = draft.report_markdown
        self._emit("writer", StepStatus.COMPLETED.value, step.summary)
        return draft

    def _stage_critic(
        self,
        state: ResearchState,
        agent: BaseAgent,
        topic: str,
        source_index: str,
        report: str,
    ) -> Optional[CriticOutput]:
        step = state.add_step(
            AgentStep(
                name="critic",
                display_name=agent.display_name,
                status=StepStatus.RUNNING.value,
                input_summary=f"draft report ({len(report)} chars) against {len(state.sources)} source(s)",
                started_at=_now(),
            )
        )
        self._emit("critic", StepStatus.RUNNING.value, "Reviewing the draft report")
        started = time.perf_counter()
        try:
            feedback: CriticOutput = agent.run(  # type: ignore[attr-defined]
                topic=topic, report=report, source_index=source_index
            )
        except (AgentError, ConfigurationError) as exc:
            step.status = StepStatus.FAILED.value
            step.error = f"Review failed: {exc}"
            step.duration_s = time.perf_counter() - started
            state.add_warning(f"The Critic agent failed to review the report: {exc}")
            self._emit("critic", StepStatus.FAILED.value, str(exc))
            return None

        step.status = StepStatus.COMPLETED.value
        step.duration_s = time.perf_counter() - started
        step.summary = (
            f"quality {feedback.quality_score}/10, {len(feedback.issues)} issue(s), "
            f"significant={feedback.significant_issues}"
        )
        step.output = feedback.model_dump()
        state.critic_feedback = feedback.model_dump()
        self._emit("critic", StepStatus.COMPLETED.value, step.summary)
        return feedback

    def _stage_revision(
        self,
        state: ResearchState,
        agent: BaseAgent,
        *,
        topic: str,
        source_index: str,
        evidence_block: str,
        draft: WriterOutput,
        critic: CriticOutput,
    ) -> None:
        """Run the single allowed revision pass."""
        step = state.add_step(
            AgentStep(
                name="writer_revision",
                display_name=f"{agent.display_name} (revision)",
                status=StepStatus.RUNNING.value,
                input_summary=f"draft + {len(critic.issues)} critic issue(s)",
                started_at=_now(),
            )
        )
        self._emit("revision", StepStatus.RUNNING.value, "Revising the report once, per critic feedback")
        started = time.perf_counter()

        if state.revision_count >= MAX_REVISIONS:
            # Defensive: this should be unreachable, but never loop.
            step.status = StepStatus.SKIPPED.value
            step.summary = "revision budget already used"
            state.add_warning("Skipped a revision request because the revision budget was used.")
            self._emit("revision", StepStatus.SKIPPED.value, step.summary)
            return

        try:
            revised: WriterOutput = agent.run(  # type: ignore[attr-defined]
                topic=topic,
                source_index=source_index,
                evidence_block=evidence_block,
                revision=True,
                previous_report=draft.report_markdown,
                critic_feedback=self._format_critic_feedback(critic),
            )
        except (AgentError, ConfigurationError) as exc:
            step.status = StepStatus.FAILED.value
            step.error = f"Revision failed: {exc}"
            step.duration_s = time.perf_counter() - started
            state.add_warning(f"The revision pass failed; the original draft is shown: {exc}")
            self._emit("revision", StepStatus.FAILED.value, str(exc))
            return

        state.revision_count += 1
        state.revised_report = revised.report_markdown
        step.status = StepStatus.COMPLETED.value
        step.duration_s = time.perf_counter() - started
        step.summary = f"revision {state.revision_count}/{MAX_REVISIONS} produced ({len(revised.report_markdown)} chars)"
        step.output = revised.model_dump()
        self._emit("revision", StepStatus.COMPLETED.value, step.summary)

    # -- finalisation -------------------------------------------------------------------
    @staticmethod
    def _format_critic_feedback(critic: CriticOutput) -> str:
        lines = [
            f"Overall assessment: {critic.overall_assessment or '(none)'}",
            f"Quality score: {critic.quality_score}/10",
            "Issues:",
        ]
        for issue in critic.issues:
            lines.append(
                f"- [{issue.severity}] {issue.category}: {issue.description} "
                f"(location: {issue.location or 'n/a'}) -> {issue.suggestion or 'no suggestion'}"
            )
        if not critic.issues:
            lines.append("- none")
        if critic.unsupported_claims:
            lines.append("Unsupported claims:")
            lines.extend(f"- {claim}" for claim in critic.unsupported_claims)
        if critic.revision_suggestions:
            lines.append("Suggested revisions:")
            lines.extend(f"- {item}" for item in critic.revision_suggestions)
        return "\n".join(lines)

    #: Headings a model-written source list might use; they get replaced by the real urls.
    _SOURCES_HEADING_RE = re.compile(
        r"(?im)^\s*#{1,6}\s*(?:sources?|references?|bibliography|works\s+cited|"
        r"参考资料|参考来源|来源|参考文献)\s*$"
    )

    @classmethod
    def _drop_model_sources_section(cls, report: str) -> tuple[str, bool]:
        """Cut a model-written sources/bibliography section so real urls replace it."""
        match = cls._SOURCES_HEADING_RE.search(report or "")
        if match is None:
            return report, False
        return report[: match.start()].rstrip(), True

    @staticmethod
    def _ensure_title(report: str, title: str) -> str:
        stripped = report.lstrip()
        if stripped.startswith("#"):
            return report
        heading = title.strip() or "Research Report"
        return f"# {heading}\n\n{report.lstrip()}"

    def _finalise_report(self, state: ResearchState, sources: Sequence[SearchResult]) -> None:
        body = state.revised_report or state.draft_report
        valid_ids = {source.source_id for source in sources}

        body, removed_sources_section = self._drop_model_sources_section(body)
        if removed_sources_section:
            state.add_warning(
                "The draft contained its own source list; it was replaced with the verified links."
            )

        body, unknown = strip_unknown_citations(body, valid_ids)
        if unknown:
            state.add_warning(
                "Removed citation(s) that pointed to sources which were never retrieved: "
                + ", ".join(unknown)
            )

        # A model can also paste a link it "remembers": keep only retrieved urls.
        body, invented_urls = sanitize_report_urls(body, {source.url for source in sources})
        if invented_urls:
            state.add_warning(
                "Removed url(s) that were never retrieved: " + ", ".join(invented_urls)
            )

        title = ""
        for step_name in ("writer_revision", "writer"):
            step = state.step(step_name)
            if step and step.output and step.output.get("title"):
                title = str(step.output["title"])
                break
        body = self._ensure_title(body, title)

        state.citations = find_citations(body)
        if not state.citations:
            state.add_warning(
                "The report contains no inline citations; treat its statements as "
                "unverified rather than source-backed."
            )
        uncited = [source_id for source_id in valid_ids if source_id not in state.citations]
        if uncited:
            state.add_warning("Source(s) not cited in the final report: " + ", ".join(sorted(uncited)))

        state.final_report = f"{body}\n\n{render_sources_section(sources)}".strip()

    def _finish(self, state: ResearchState, *, failed: bool = False) -> ResearchState:
        if failed:
            state.status = RunStatus.FAILED.value
        elif state.warnings or any(
            step.status in (StepStatus.FAILED.value, StepStatus.SKIPPED.value) for step in state.steps
        ):
            state.status = RunStatus.COMPLETED_WITH_WARNINGS.value
        else:
            state.status = RunStatus.COMPLETED.value

        state.finished_at = _now()
        self._emit(
            "pipeline",
            state.status,
            "Research finished" if state.succeeded else "Research did not produce a report",
        )
        return state


def run_research(topic: str, *, pipeline: Optional[ResearchPipeline] = None, **kwargs: Any) -> ResearchState:
    """Convenience wrapper: build a pipeline (if needed) and research ``topic``.

    ``kwargs`` are passed to :meth:`ResearchPipeline.from_env`, so settings such as
    ``max_sources``, ``max_pages``, ``llm``, ``search_fn`` or ``fetch_fn`` can be supplied
    and override the environment defaults.
    """
    active = pipeline or ResearchPipeline.from_env(**kwargs)
    return active.run(topic)


__all__ = [
    "AgentStep",
    "MAX_REVISIONS",
    "ResearchPipeline",
    "ResearchState",
    "RunStatus",
    "StepStatus",
    "run_research",
]
