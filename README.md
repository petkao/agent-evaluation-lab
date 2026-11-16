# Agent Evaluation Lab

A hands-on Python lab for evaluating **LangGraph agents with RAGAS**, curating governed golden datasets, measuring tool-calling reliability, and enforcing quality gates before release.

This repository develops an evaluation workflow incrementally: from individual RAGAS metrics to a reusable evaluation package, AI-assisted test-case generation, human-approved golden datasets, repeat-run reliability analysis, and a local CI-style gate.

> **Project status:** Lessons 1–8.7.3 have been exercised locally. The local quality gate and its negative tests pass. Git-commit-bound evaluation artifacts and GitHub Actions integration are planned next; they are **not yet implemented**.

## What the lab demonstrates

- **LangGraph execution:** A travel-assistant agent routes requests to `search_hotels` and `get_weather` tools.
- **RAGAS agent evaluation:** Tool-call accuracy and tool-call F1, with earlier exercises exploring answer-level and RAG evaluation metrics.
- **Reusable evaluation infrastructure:** Adapters, evaluators, result models, JSONL storage, and dataset loaders.
- **Golden dataset governance:** Synthetic candidate generation, structural and capability validation, semantic review, provenance hashes, human batch approval, and promotion to an approved dataset.
- **Regression detection:** Compare matching test cases and metrics between saved evaluation runs.
- **Repeated-run reliability:** Calculate per-test pass rates, descriptive statistics, and category summaries across executions.
- **Local release gate:** Verify dataset approval and integrity, evaluation completeness, and score thresholds; fail with a nonzero exit status when checks do not pass.

## Architecture

```text
                         Test-case development
                    ┌────────────────────────────┐
                    │ Manually authored cases    │
                    │ + synthetic candidates     │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Structural/capability      │
                    │ validation + LLM review    │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Human batch approval       │
                    │ + SHA-256 provenance       │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Approved Golden Dataset v2 │
                    └─────────────┬──────────────┘
                                  │
             ┌────────────────────▼──────────────────┐
             │ LangGraph travel agent                │
             │ search_hotels / get_weather           │
             └────────────────────┬──────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ LangGraph → RAGAS adapter  │
                    │ ToolCallAccuracy / F1      │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Versioned JSONL run data  │
                    │ run_id / dataset SHA-256   │
                    └─────────────┬──────────────┘
                                  │
                 ┌────────────────┴─────────────────┐
                 ▼                                  ▼
          Regression comparison              Reliability analysis
                 └────────────────┬─────────────────┘
                                  ▼
                           Local CI gate
                         PASS / FAIL exit
```

## Technology stack

| Component | Technology |
|---|---|
| Language | Python 3.13 (local lab environment) |
| Dependency management | Poetry |
| Agent orchestration | LangGraph and LangChain |
| LLM integration | `langchain-openai` (`gpt-4o-mini` in the exercises) |
| Evaluation | RAGAS |
| Schemas and validation | Pydantic |
| Configuration | `python-dotenv` |
| Data and results | JSON / JSONL |
| Version control | Git and GitHub |

Versions observed during the lab include RAGAS 0.4.3 and LangGraph 1.2.21. The project's `pyproject.toml` and lockfile, when present, are the authoritative dependency specifications.

## Repository layout

The following are the principal files developed during the lessons; earlier exercises may add more scripts.

```text
agent-evaluation-lab/
├── evaluation/
│   ├── adapters.py                 # LangGraph → RAGAS messages
│   ├── tool_evaluator.py           # Tool-call accuracy and F1
│   ├── models.py                   # Structured evaluation records
│   ├── storage.py                  # JSONL persistence
│   ├── dataset_loader.py           # Golden dataset loader
│   ├── synthetic_generator.py      # Synthetic candidate generation
│   ├── candidate_validator.py      # Candidate and coverage checks
│   ├── semantic_reviewer.py        # LLM-based semantic review
│   ├── provenance.py               # Canonical candidate hashing
│   ├── batch_approval.py           # Batch review manifest
│   ├── golden_promoter.py          # Approved dataset promotion
│   ├── regression_gate.py          # Run-to-run metric comparison
│   ├── reliability.py              # Repeated-run analysis
│   ├── ci_gate.py                  # Local CI quality gate
│   └── git_provenance.py           # Git commit/worktree inspection
├── datasets/
│   ├── hotel_golden_v1.jsonl
│   ├── hotel_golden_v2.jsonl
│   ├── hotel_golden_v2_approval.json
│   └── hotel_synthetic_candidates_v1.jsonl
├── results/                        # Generated locally; gitignored
├── lesson8_5_1_evaluate_golden_v2.py
├── lesson8_5_2_regression_gate.py
├── lesson8_5_3_compare_runs.py
├── lesson8_6_2_analyze_reliability.py
├── lesson8_7_1_ci_gate.py
├── lesson8_7_2_test_ci_failures.py
├── lesson8_7_3_git_provenance.py
├── pyproject.toml
└── README.md
```

## Setup

### Prerequisites

- Python compatible with the project's `pyproject.toml` (the lab was run with Python 3.13.2)
- Poetry
- An OpenAI API key for exercises that invoke the LLM

```bash
# From the repository root
poetry install
```

Create a local `.env` file containing your key:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

**Never commit `.env` or actual credentials.** The repository's `.gitignore` excludes `.env`, generated `results/`, Python caches, and virtual environments.

> Some exercises make paid LLM API calls. Dataset verification, regression comparisons, reliability summaries, and stored-result gate checks do not themselves require new LLM calls.

## Golden datasets

### v1: Manually authored cases

`datasets/hotel_golden_v1.jsonl` contains three introductory cases covering hotel search, current-weather lookup, and tool selection.

### v2: Approved synthetic cases

`datasets/hotel_golden_v2.jsonl` contains **10 approved cases**:

| Category | Cases |
|---|---:|
| Hotel search only | 4 |
| Current weather only | 3 |
| Combined hotel search and weather | 3 |
| **Total** | **10** |

The approved dataset is accompanied by `datasets/hotel_golden_v2_approval.json`, which records approval metadata and a SHA-256 digest of the golden file.

The synthetic-data workflow uses multiple safeguards: tool-schema checks, supported-capability checks, coverage requirements, LLM semantic review, candidate content hashes, and a human approval step. LLM review is **assistance**, not a substitute for human approval.

Verify the approved dataset:

```bash
poetry run python lesson8_4_3_verify_golden.py
```

Expected final line:

```text
Verification PASSED
```

**Tool limitations in this lab:** The sample tools support hotel search under an inclusive price ceiling and **current** weather lookup. Future or historical weather requests, booking actions, and travel-date reservations are not supported positive test cases. Unsupported-capability tests are a possible future negative-test suite.

## Run evaluations

Evaluate all 10 approved cases against the LangGraph agent:

```bash
poetry run python lesson8_5_1_evaluate_golden_v2.py
```

This appends run-level records to `results/golden_evaluations_v2.jsonl` and writes a latest-run summary to `results/golden_summary_v2.json`.

Each evaluation record includes a `run_id`, test ID, agent version, dataset version, dataset SHA-256, category, tool-call accuracy, tool-call F1, and evaluation timestamp.

**Metric scope:** Perfect tool-call accuracy and F1 mean the expected tools and arguments were selected according to these metrics. They do **not** establish that returned hotel/weather data are factually accurate, that final responses are useful, or that the agent generalizes to arbitrary requests.

### Compare actual runs

After collecting at least two complete runs:

```bash
poetry run python lesson8_5_3_compare_runs.py
```

This compares the earliest and latest saved runs by test ID and metric, checking that their dataset hashes match. The separate regression-gate self-test is:

```bash
poetry run python lesson8_5_2_regression_gate.py
```

### Analyze repeated-run reliability

```bash
poetry run python lesson8_6_2_analyze_reliability.py
```

The analyzer checks for duplicate and incomplete runs and reports per-test pass rates, mean/min/max metric scores, sample standard deviation, and category summaries. It saves `results/reliability_summary_v2.json`.

**Observed lab results:** Five complete runs of the same 10-case approved dataset yielded **50/50 successful test executions**, with tool-call accuracy and F1 both 1.000 in every recorded execution. This is an observed result for a small, repeated test set—not a guarantee of 100% reliability or a statistically independent 50-case generalization study.

## Local CI quality gate

Run the stored-result quality gate:

```bash
poetry run python lesson8_7_1_ci_gate.py
```

The current gate verifies:

1. The golden dataset has approved status.
2. The golden dataset's SHA-256 matches the approval audit.
3. The latest stored evaluation run has the complete expected set of test IDs, with no duplicates.
4. Each evaluation record refers to the approved dataset hash.
5. Every tool-call metric meets the configured minimum (currently **1.000**).

A passing gate returns exit code `0`; a rejected gate returns a nonzero exit code.

Test that the gate rejects invalid results:

```bash
poetry run python lesson8_7_2_test_ci_failures.py
```

The negative tests deliberately introduce a degraded metric, an incomplete run, and a dataset-hash mismatch **in temporary copies**. All three were correctly rejected in the lab.

**Security/release limitation:** This gate currently validates **stored** results. It does not prove that those results came from the current checkout or commit. Do not use it alone to authorize production deployment. A future CI workflow must run the candidate agent on the target revision, record code provenance, and validate the resulting artifacts.

## Git provenance

Inspect the current commit and worktree state:

```bash
poetry run python lesson8_7_3_git_provenance.py
```

The helper in `evaluation/git_provenance.py` can also enforce a clean worktree with `get_git_provenance(require_clean=True)`. The existing lesson script uses `require_clean=False` for inspection.

**Planned:** Record the Git SHA on each new evaluation record and verify that the candidate results match the revision being evaluated. GitHub Actions automation will follow.

## Lesson progression

| Lessons | Focus |
|---|---|
| 1–4 | RAGAS fundamentals and RAG evaluation metrics |
| 5 | LangGraph travel agent and tool trajectory |
| 6 | Agent goal accuracy, tool-call metrics, and adapters |
| 7 | Reusable evaluation package and JSONL persistence |
| 8.1–8.2 | Golden Dataset v1 and evaluation runner |
| 8.3 | Synthetic test generation and validation |
| 8.4 | Semantic review, provenance, human approval, and Golden Dataset v2 |
| 8.5 | Baseline evaluation and regression detection |
| 8.6 | Repeated-run reliability analysis |
| 8.7.1–8.7.3 | Local CI gate, failure tests, and Git provenance inspection |
| Next | Commit-aware evaluations, candidate comparison, GitHub Actions |

## Engineering considerations and future work

- Tie every evaluation artifact to the exact Git commit, agent configuration, model, and prompt version.
- Require complete runs and fail closed on missing, malformed, non-finite, or out-of-range metric values.
- Use an explicit, immutable baseline run or release reference instead of relying solely on the first saved run.
- Separate deterministic unit tests from API-dependent integration evaluations; avoid requiring live LLM calls on every trivial change.
- Add negative-capability, tool-failure, prompt-injection, and adversarial test suites.
- Expand evaluation beyond tool selection to answer quality, groundedness, safety, latency, cost, and operational reliability.
- Use more diverse and independently sampled test cases before making strong statistical reliability claims.
- Implement GitHub Actions with explicit permissions, secret handling, and branch-protection rules.

## Reproducibility and data handling

Approved datasets and their audit metadata belong in version control. Generated run histories and summaries live under `results/` and are ignored by Git. Preserve important baseline run artifacts in a controlled, immutable artifact store when moving to production CI; local ignored files are not available automatically to a fresh GitHub Actions runner.

---

**Project:** [agent-evaluation-lab](https://github.com/petkao/agent-evaluation-lab)  
**Purpose:** Hands-on, production-oriented evaluation engineering for LLM agents.
