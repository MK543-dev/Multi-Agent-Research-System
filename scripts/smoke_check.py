"""Optional integration smoke check (no API keys, no network).

The offline unit tests replace the language model entirely. This script additionally
verifies the two pieces that depend on *installed* library versions:

1. the real ``ChatOpenAI`` client can be constructed and can bind every structured-output
   schema used by the agents (catches LangChain API drift), and
2. the Streamlit app actually renders - headlessly, through Streamlit's own
   ``AppTest`` - both for a completed run and for the missing-API-key path.

It uses obviously fake placeholder credentials, derived at runtime, purely because the
OpenAI client refuses to construct without an environment variable present. No request is
ever sent.

Usage (from the project root, inside your virtual environment)::

    python scripts/smoke_check.py

Exit code 0 means every check passed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def placeholder(prefix: str, *parts: str) -> str:
    """Build a clearly fake credential at runtime (never a stored literal)."""
    return prefix + "-".join(parts)


failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


# --------------------------------------------------------------------------------------
# 1. Real ChatOpenAI plumbing
# --------------------------------------------------------------------------------------
os.environ.setdefault("OPENAI_API_KEY", placeholder("sk-", "placeholder", "value", "not", "real"))
os.environ.setdefault("TAVILY_API_KEY", placeholder("tvly-", "placeholder", "value"))

from agents import (  # noqa: E402
    CriticOutput,
    ReaderOutput,
    SearchQueries,
    WriterOutput,
    build_chat_model,
)

try:
    llm = build_chat_model()
    check("ChatOpenAI constructs with the installed langchain-openai", True, type(llm).__name__)
    model_name = str(getattr(llm, "model_name", getattr(llm, "model", "")))
    check("default model is gpt-4o-mini", "gpt-4o-mini" in model_name, model_name)
    for schema in (SearchQueries, ReaderOutput, WriterOutput, CriticOutput):
        bound = llm.with_structured_output(schema)
        check(f"with_structured_output({schema.__name__}) binds", bound is not None, type(bound).__name__)
except Exception as exc:  # noqa: BLE001
    check("ChatOpenAI plumbing", False, f"{exc.__class__.__name__}: {exc}")

# --------------------------------------------------------------------------------------
# 2. Streamlit app, headless
# --------------------------------------------------------------------------------------
from pipeline import ResearchPipeline  # noqa: E402
from tests.test_system import FakeLLM, default_responder, make_fetch, make_search  # noqa: E402

state = ResearchPipeline(
    llm=FakeLLM(default_responder),
    search_fn=make_search(),
    fetch_fn=make_fetch(),
    max_sources=3,
    max_pages=3,
).run("UI smoke topic")
check("mocked pipeline run completes", state.status == "completed", f"status={state.status}")

try:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.session_state["last_state"] = state
    at.run()
    errors = [str(e.value) for e in at.exception]
    check("app renders a completed run without raising", not errors, "; ".join(errors))
    print(f"       markdown blocks rendered: {len(at.get('markdown'))}")
    print(f"       tab elements: {len(at.get('tabs'))}")

    # Revision path: a critic that always flags issues must render draft/revised tabs.
    from agents import CriticIssue, CriticOutput

    def always_critical(schema, messages):
        if schema is CriticOutput:
            return CriticOutput(
                overall_assessment="Needs work.",
                significant_issues=True,
                quality_score=4,
                issues=[CriticIssue(severity="high", category="citation", description="Uncited claim")],
                unsupported_claims=["Everything"],
                revision_suggestions=["Ground every claim."],
            )
        return default_responder(schema, messages)

    revised_state = ResearchPipeline(
        llm=FakeLLM(always_critical),
        search_fn=make_search(),
        fetch_fn=make_fetch(),
        max_sources=3,
        max_pages=3,
    ).run("UI smoke topic")
    check("mocked run performs exactly one revision", revised_state.revision_count == 1)

    at3 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at3.session_state["last_state"] = revised_state
    at3.run()
    errors3 = [str(e.value) for e in at3.exception]
    check("app renders the revision view without raising", not errors3, "; ".join(errors3))
    tabs = len(at3.get("tabs"))
    check("revision view exposes draft and revised report", tabs >= 2, f"tabs={tabs}")

    # Missing-key path, exercised through the real app code and widgets.
    os.environ.pop("OPENAI_API_KEY", None)
    at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at2.run()
    before = [e.value for e in at2.error]
    check("missing key is flagged before starting", any("OPENAI_API_KEY" in m for m in before), str(before))
    at2.text_input[0].set_value("A topic with no key configured").run()
    at2.button[0].click().run()
    after = [e.value for e in at2.error]
    check(
        "clicking start without a key shows the error",
        any("OPENAI_API_KEY" in m for m in after),
        str(after),
    )
except ImportError as exc:  # pragma: no cover - streamlit always ships AppTest
    check("streamlit AppTest available", False, str(exc))

print()
print("SMOKE RESULT:", "ALL PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}")
sys.exit(1 if failures else 0)
