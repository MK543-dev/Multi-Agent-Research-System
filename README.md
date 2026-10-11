# Multi-Agent Research System

An autonomous research assistant built with **Python + LangChain + GPT-4o-mini**.
Four specialised agents cooperate on a topic you type in, and every important claim in the
final report is tied to a source that was actually downloaded.

```
                 ┌──────────────┐   queries   ┌───────────────┐
  topic ───────► │ Search Agent │ ──────────► │ Tavily search │
                 └──────┬───────┘             └───────────────┘
                        │ numbered sources (S-1, S-2, …)
                        ▼
                 ┌──────────────┐   fetches   ┌──────────────────┐
                 │ Reader Agent │ ──────────► │ requests + bs4   │
                 └──────┬───────┘             └──────────────────┘
                        │ structured notes, gaps, conflicts
                        ▼
                 ┌──────────────┐   draft     ┌───────────────────────────┐
                 │ Writer Agent │ ◄────────── │ Critic Agent              │
                 └──────┬───────┘  feedback   │ (max 1 revision, then stop)│
                        │  └────────────────►└───────────────────────────┘
                        ▼
        final report + verified source urls + per-agent audit trail
```

## Features

| Requirement | How it is met |
|---|---|
| Four specialised agents | `agents.py`: `SearchAgent`, `ReaderAgent`, `WriterAgent`, `CriticAgent` — one responsibility each |
| Source-grounded claims | Sources get ids (`S-1`, `S-2`, …); the Writer may only cite those ids; citations that were never retrieved are stripped and reported; the source list with real, clickable URLs is appended by the application, never written by the model |
| Critic + revision loop | The Critic returns structured feedback (assessment, issues, unsupported claims, suggestions). Significant issues trigger **exactly one** Writer revision, hard-capped by `pipeline.MAX_REVISIONS` |
| Transparent orchestration | `ResearchState` records each agent's status, input summary, structured output, duration, and errors; the UI shows all of it plus draft vs. revised report |
| Reliability | Per-request timeouts, page-size and length caps, source limits, retries on malformed model output, graceful degradation, and clear error messages |
| Untrusted web content | Page text is stripped of scripts/markup, scanned for instruction-like phrasing, wrapped in `<untrusted_source>` delimiters, and every prompt states that such text is data — never commands |
| No exposed secrets | Keys are read from the environment only; nothing is logged, displayed, or embedded in code |

## Project structure

```
.
├── agents.py          # The four agents + Pydantic output models + LLM plumbing
├── tools.py           # Search (Tavily) and page retrieval (requests + BeautifulSoup)
├── pipeline.py        # Orchestration, shared ResearchState, revision cap, final assembly
├── app.py             # Streamlit interface
├── tests/
│   └── test_system.py # 50 offline tests (mocked LLM, mocked search/fetch, no API keys)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Requirements

* Python **3.10+** (developed and verified on 3.13)
* An OpenAI API key (for `gpt-4o-mini`)
* A Tavily API key — free tier at <https://app.tavily.com/home>

## Setup

### 1. Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your keys

```powershell
Copy-Item .env.example .env      # macOS/Linux: cp .env.example .env
```

Open `.env` and fill in:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

`.env` is git-ignored; `.env.example` only ever contains placeholders.

Optional overrides (defaults shown):

```
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0
MAX_SOURCES=5
MAX_PAGES=4
MAX_PAGE_CHARS=6000
REQUEST_TIMEOUT=15
```

## Run the app

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>). Type a focused research
question, press **🚀 Start research**, and watch each agent report its progress:

1. **Agent activity** — live status per agent.
2. **Final report** — the grounded Markdown report, with a download button.
3. **Sources** — every retrieved URL as a clickable link, including pages that failed.
4. **What changed** — original draft and revised report side by side (when a revision ran).
5. **Critic review** — quality score, issues, unsupported claims, suggestions.
6. **Agent outputs** — each agent's structured output plus the raw run state (JSON).

## Use it without the UI

```python
from pipeline import run_research

state = run_research("How are solid-state batteries being commercialised in 2026?")
print(state.status)          # completed | completed_with_warnings | failed
print(state.final_report)    # report + verified source list
print(state.summary_lines()) # one audit line per agent
```

## How a run works

1. **Search Agent** asks the model for up to three complementary queries, runs them,
   de-duplicates by URL, keeps `MAX_SOURCES` hits and numbers them `S-1…S-n`.
2. **Reader Agent** downloads up to `MAX_PAGES` of those pages (timeouts, size caps,
   `MAX_PAGE_CHARS` truncation), then extracts one note per source with the source id and
   URL attached, plus explicit *gaps* and *conflicts*.
3. **Writer Agent** drafts the report (Introduction, Key Findings, Analysis, Conclusion,
   plus Limitations when the evidence is thin) citing only the supplied ids.
4. **Critic Agent** reviews it and returns structured feedback.
5. **Revision** happens once, and only if the Critic flags significant issues. If that
   revision fails, the draft is shown and the failure is reported.
6. **Assembly** strips any citation pointing at a source that was never retrieved,
   replaces any model-written source list with the real URLs, and records warnings in
   `state.warnings`.

Every run costs 4–5 model calls (query planning, reading, drafting, reviewing, optional
revision).

## Error handling

| Situation | Behaviour |
|---|---|
| Missing `OPENAI_API_KEY` / `TAVILY_API_KEY` | Run fails immediately with a message naming the missing variable; the UI shows it before starting |
| Empty topic | Rejected before any API call |
| Search API error or zero results | Run fails with the API's message |
| One page fails to load | Warning listing the URL; the pipeline continues with the pages that worked |
| Every page fails | Run fails with “No usable sources …” instead of inventing a report |
| Model returns malformed output | Retried once, then a JSON fallback; the Reader degrades to raw excerpts, the Critic degrades to “no review”, the Writer fails loudly |
| Report cites an unknown source | The citation is removed and reported as a warning |
| Critic demands endless revisions | Impossible: `MAX_REVISIONS = 1`, clamped even if a caller passes more |
| Progress callback raises | Ignored — the run continues |

## Tests

```bash
python -m unittest discover -s tests -t .        # add -v for per-test names
```

50 tests, all offline: the model, the search API and page fetching are replaced by
deterministic fakes (HTTP-level tests patch `requests.get`). They cover the happy path,
state and citation bookkeeping, every failure mode in the table above, and the
one-revision cap.

## Design notes and limitations

* Retrieval is sequential and capped, so a run stays within a predictable time and cost
  budget rather than scraping an unbounded number of pages.
* The Critic judges grounding against the notes and source list; it does not re-fetch
  pages, so it can still miss subtle misreadings of a source.
* Prompts are written in English for model reliability; the report itself is produced in
  the language of the topic.
* Reports are AI-generated summaries of web sources. They cite their sources so you can
  check them — treat them as a research starting point, not as verified fact.

## Security notes

* Retrieved pages are untrusted input: markup is stripped, instruction-like phrases are
  flagged in `page.flagged_phrases` and shown in the UI, and the text is wrapped in
  explicit data delimiters before reaching the model.
* No credentials are read from user input, logged, or rendered. `.env` is git-ignored.
* Only `http`/`https` URLs are fetched, response size is capped, and non-text content
  types are rejected.
