"""Streamlit interface for the multi-agent research system.

Run with::

    streamlit run app.py

The page lets the user type a topic, watch each agent report its progress, then read the
final report with clickable sources, the critic's feedback, the original draft next to the
revised one, and every agent's raw structured output. API keys are only ever read from the
environment; they are never displayed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Load .env before importing project modules so that environment-driven defaults
# (model name, source limits) are already visible when they are first read.
load_dotenv()

from agents import ConfigurationError  # noqa: E402  (import after load_dotenv)
from pipeline import ResearchPipeline, ResearchState, RunStatus, StepStatus  # noqa: E402

APP_TITLE = "Multi-Agent Research System"

#: Pipeline stages shown in the progress panel: (stage key, label, icon budget)
STAGES: List[Tuple[str, str]] = [
    ("search", "Search Agent"),
    ("reader", "Reader Agent"),
    ("writer", "Writer Agent"),
    ("critic", "Critic Agent"),
    ("revision", "Writer revision (max 1)"),
]

STATUS_ICON = {
    StepStatus.PENDING.value: "⚪",
    StepStatus.RUNNING.value: "🔄",
    StepStatus.COMPLETED.value: "✅",
    StepStatus.FAILED.value: "❌",
    StepStatus.SKIPPED.value: "⏭️",
}


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
def render_sidebar() -> Tuple[int, int]:
    """Configuration panel. Returns ``(max_sources, max_pages)``."""
    with st.sidebar:
        st.header("Configuration")

        st.subheader("API keys")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        if openai_key:
            st.success("OPENAI_API_KEY detected", icon="✅")
        else:
            st.error("OPENAI_API_KEY missing", icon="❌")
        if tavily_key:
            st.success("TAVILY_API_KEY detected", icon="✅")
        else:
            st.error("TAVILY_API_KEY missing", icon="❌")
        st.caption("Keys are read from the environment (.env). Values are never displayed.")

        st.subheader("Run settings")
        max_sources = st.slider("Sources to retrieve", min_value=1, max_value=10, value=5)
        max_pages = st.slider("Pages to read", min_value=1, max_value=8, value=4)

        st.subheader("Model")
        st.code(os.getenv("OPENAI_MODEL", "gpt-4o-mini"), language=None)
        st.caption("Set OPENAI_MODEL in .env to switch models.")

        st.divider()
        st.caption(
            "Retrieved pages are treated as untrusted data: instruction-like text is "
            "flagged and never executed. Citations are limited to sources that were "
            "actually downloaded."
        )
    return max_sources, max_pages


# --------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------
def run_with_progress(topic: str, max_sources: int, max_pages: int) -> Optional[ResearchState]:
    """Execute the pipeline, streaming per-agent progress into the page."""
    st.subheader("Agent activity")
    progress = st.progress(0.0, text="Starting…")
    rows: Dict[str, Any] = {}
    for stage, label in STAGES:
        rows[stage] = st.empty()
        rows[stage].markdown(f"{STATUS_ICON[StepStatus.PENDING.value]} **{label}** — waiting")

    finished: set[str] = set()

    def on_progress(stage: str, status: str, message: str) -> None:
        if stage == "pipeline":
            progress.progress(1.0 if status in ("completed", "completed_with_warnings", "failed") else 0.0,
                              text=message)
            return
        label = dict(STAGES).get(stage, stage)
        icon = STATUS_ICON.get(status, "⚪")
        if stage in rows:
            rows[stage].markdown(f"{icon} **{label}** — {message}")
        if status in (StepStatus.COMPLETED.value, StepStatus.FAILED.value, StepStatus.SKIPPED.value):
            finished.add(stage)
        progress.progress(
            min(len(finished) / max(1, len(STAGES)), 1.0),
            text=f"{label}: {message}",
        )

    pipeline = ResearchPipeline(
        max_sources=max_sources,
        max_pages=max_pages,
        progress_callback=on_progress,
    )
    try:
        return pipeline.run(topic)
    except Exception as exc:  # last-resort guard: the UI must stay usable
        st.error(f"Unexpected failure while running the pipeline: {exc.__class__.__name__}: {exc}")
        return None


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------
def render_status_banner(state: ResearchState) -> None:
    if state.status == RunStatus.FAILED.value:
        st.error("Research failed — see the messages below.")
    elif state.status == RunStatus.COMPLETED_WITH_WARNINGS.value:
        st.warning("Research finished with warnings. Review them before trusting the report.")
    elif state.status == RunStatus.COMPLETED.value:
        st.success("Research finished.")
    else:
        st.info(f"Run status: {state.status}")

    meta = f"Topic: **{state.topic}** · started {state.started_at} · finished {state.finished_at}"
    if state.revision_count:
        meta += f" · revisions: {state.revision_count}/1"
    st.caption(meta)

    for message in state.errors:
        st.error(message)
    for message in state.warnings:
        st.warning(message)


def render_sources(state: ResearchState) -> None:
    st.subheader("Sources")
    if not state.sources:
        st.info("No sources were collected.")
        return
    for source in state.sources:
        title = source.get("title") or source.get("url", "")
        url = source.get("url", "")
        st.markdown(f"- **[{source.get('source_id', '?')}]** [{title}]({url})")
    pages = state.usable_pages
    st.caption(
        f"{len(pages)} of {len(state.pages)} page(s) were readable"
        if state.pages
        else "Page retrieval did not run."
    )
    with st.expander("Retrieved page details (including failures)"):
        for page in state.pages:
            status = "readable" if page.get("ok") else f"not readable — {page.get('error')}"
            line = f"**[{page.get('source_id') or '?'}]** {page.get('url')} — {status}"
            if page.get("truncated"):
                line += " (truncated)"
            st.markdown(line)
            if page.get("flagged_phrases"):
                st.warning(
                    "Instruction-like text was found in this page and treated as data: "
                    + " | ".join(page["flagged_phrases"])
                )


def render_report(state: ResearchState) -> None:
    st.subheader("Final report")
    if not state.final_report:
        st.info("No report was produced.")
        return

    st.markdown(state.final_report)
    st.download_button(
        "⬇️ Download report (Markdown)",
        data=state.final_report,
        file_name="research_report.md",
        mime="text/markdown",
    )
    if state.citations:
        st.caption("Inline citations used: " + ", ".join(state.citations))
    else:
        st.caption("No inline citations were produced; treat the text as unverified.")


def render_critic(state: ResearchState) -> None:
    feedback = state.critic_feedback
    st.subheader("Critic review")
    if not feedback:
        st.info("The Critic agent did not return feedback.")
        return

    score = feedback.get("quality_score", "n/a")
    st.markdown(
        f"**Quality score:** {score}/10 · "
        f"**Significant issues:** {'yes' if feedback.get('significant_issues') else 'no'} · "
        f"**Revisions applied:** {state.revision_count}/1"
    )
    if feedback.get("overall_assessment"):
        st.write(feedback["overall_assessment"])

    issues = feedback.get("issues") or []
    if issues:
        st.markdown("**Issues found**")
        for issue in issues:
            st.markdown(
                f"- `{issue.get('severity', 'medium')}` / {issue.get('category', 'other')}: "
                f"{issue.get('description', '')}"
                + (f" _(at: {issue['location']})_" if issue.get("location") else "")
                + (f" → {issue['suggestion']}" if issue.get("suggestion") else "")
            )
    if feedback.get("unsupported_claims"):
        st.markdown("**Unsupported claims**")
        for claim in feedback["unsupported_claims"]:
            st.markdown(f"- {claim}")
    if feedback.get("revision_suggestions"):
        st.markdown("**Suggested revisions**")
        for item in feedback["revision_suggestions"]:
            st.markdown(f"- {item}")


def render_revision_diff(state: ResearchState) -> None:
    if not state.revised_report:
        st.caption("No revision was required: the draft passed the critic's review.")
        return
    st.subheader("What changed")
    original, revised = st.tabs(["Original draft", "Revised report"])
    with original:
        st.markdown(state.draft_report)
    with revised:
        st.markdown(state.revised_report)
    st.caption(
        "The revised report is the one shown under “Final report”. The verified source "
        "list is appended to it automatically."
    )


def render_agent_outputs(state: ResearchState) -> None:
    st.subheader("Agent outputs")
    st.markdown(
        "\n".join(f"- {line}" for line in state.summary_lines())
        or "- no agent ran"
    )

    for step in state.steps:
        icon = STATUS_ICON.get(step.status, "⚪")
        header = f"{icon} {step.display_name} — {step.summary or step.error or step.status}"
        with st.expander(header, expanded=False):
            st.markdown(f"**Status:** `{step.status}` · **Duration:** {step.duration_s:.2f}s")
            if step.input_summary:
                st.markdown(f"**Input:** {step.input_summary}")
            if step.error:
                st.error(step.error)
            if step.output is not None:
                _render_structured_output(step.name, step.output)


def _render_structured_output(step_name: str, output: Dict[str, Any]) -> None:
    """Human-readable rendering of a structured agent output, with raw JSON available."""
    if step_name == "reader":
        notes = output.get("notes") or []
        st.markdown(f"**Notes extracted:** {len(notes)} source(s), evidence quality: "
                    f"{output.get('overall_evidence_quality', 'unknown')}")
        for note in notes:
            st.markdown(f"**[{note.get('source_id') or '?'}]** {note.get('url', '')}")
            for fact in note.get("key_facts") or []:
                st.markdown(f"- {fact}")
            if note.get("limitations"):
                st.caption(f"limitation: {note['limitations']}")
        for gap in output.get("gaps") or []:
            st.markdown(f"- _gap:_ {gap}")
        for conflict in output.get("conflicts") or []:
            st.markdown(f"- _conflict:_ {conflict}")
    elif step_name in ("writer", "writer_revision"):
        st.markdown(f"**Title:** {output.get('title', '')}")
        used = output.get("used_source_ids") or []
        st.markdown(f"**Sources cited:** {', '.join(used) if used else 'none'}")
        for uncertainty in output.get("uncertainties") or []:
            st.markdown(f"- _uncertainty:_ {uncertainty}")
    elif step_name == "search":
        for source in output.get("sources") or []:
            st.markdown(
                f"- **[{source.get('source_id')}]** [{source.get('title') or source.get('url')}]"
                f"({source.get('url')})"
            )

    with st.expander("Raw structured output (JSON)"):
        st.json(output)


def render_state(state: ResearchState) -> None:
    render_status_banner(state)
    render_report(state)
    render_sources(state)
    render_revision_diff(state)
    render_critic(state)
    render_agent_outputs(state)
    with st.expander("Full run state (JSON)"):
        st.json(state.to_dict())


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
    st.title(f"🔎 {APP_TITLE}")
    st.caption("Search → Reader → Writer → Critic. LangChain + GPT-4o-mini, grounded in retrieved sources.")

    max_sources, max_pages = render_sidebar()

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. How are solid-state batteries being commercialised in 2026?",
        help="Ask a focused question — narrower topics produce better-grounded reports.",
    )
    start = st.button("🚀 Start research", type="primary")

    if start:
        if not topic.strip():
            st.error("Please enter a research topic before starting.")
        elif not os.getenv("OPENAI_API_KEY", "").strip():
            st.error(
                "OPENAI_API_KEY is not set. Copy .env.example to .env, add your key, "
                "then restart the app."
            )
        elif not os.getenv("TAVILY_API_KEY", "").strip():
            st.error(
                "TAVILY_API_KEY is not set. Copy .env.example to .env, add your Tavily "
                "key (free tier: https://app.tavily.com/home), then restart the app."
            )
        else:
            try:
                state = run_with_progress(topic.strip(), max_sources, max_pages)
            except ConfigurationError as exc:
                st.error(str(exc))
                state = None
            if state is not None:
                st.session_state["last_state"] = state

    state: Optional[ResearchState] = st.session_state.get("last_state")
    if state is None:
        st.info("Enter a topic and start a run. Each agent's work will appear here as it happens.")
        return

    st.divider()
    render_state(state)


if __name__ == "__main__":
    main()
