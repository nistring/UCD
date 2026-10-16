# UCD: a cohort-design agent for studying agentic AI on clinical data

UCD turns a clinical research question into an observational cohort study on an EHR database (MIMIC-IV 3.1 out of the box). An LLM agent writes the protocol with you, designs the cohort in SQL, runs an explainable model, and revises its design under two critics. **The language model never sees a patient row**: only table schemas, documentation and aggregate counts.

It was built for the Korea Clinical Datathon 2026 (theme *"Agentic AI: Light & Shadow"*) to observe what such an agent does well, and where it fails, when it works on clinical data.

> [!WARNING]
> **Not for clinical use.** This is a research tool for studying how agentic AI behaves when it processes clinical data. It is not a medical device. Its output has not been validated and must not inform patient care, clinical decisions or published findings unless a qualified researcher has verified it independently. The agent can and does make design errors: observing them is the point of the project.

---

## Contents
1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Running a study](#running-a-study)
5. [Configuration](#configuration)
6. [Privacy model](#privacy-model)
7. [Comparing with a ground truth](#comparing-with-a-ground-truth)
8. [Adding a dataset](#adding-a-dataset)
9. [Insight](#insight)
10. [Repository layout](#repository-layout)
11. [Development](#development)

---

## How it works

```
 question / protocol (.txt)
        │
        ▼
 [1 Intake LLM]  checks it against a standard protocol format and the dataset schema,
        │        asks you about gaps (multiple choice); you confirm or edit protocol.md
        ▼
 [2 Planner LLM] ── design/spec.yaml + design/query.sql ──► [3 Gate + Runner]  (deterministic, no LLM)
        ▲                                                         │ aggregates only
        │                                                         ▼
        └──── feedback + internal error I ◄──── [4 Rule critic + LLM clinical critic]
                    (at most 3 executed designs; stops when no blocking finding remains)
        │
        ▼
 final.md ──► [5 Evaluator] optional ground-truth CSV, compared deterministically, never fed back
```

| Step | What it does |
|---|---|
| **1. Intake** | Checks your request for objective, population, exclusions, time zero, exposure, outcome, covariates, analysis and primary estimand. It sees the dataset's schema, so it only offers choices the data can measure. The protocol you confirm becomes the gold standard for the rest of the run. |
| **2. Planner** | Explores the data through aggregate-only queries and dictionary lookups, then writes two files: a design spec (population flags, time zero, windows, outcome, model) and the SQL that builds one row per unit. It edits them by line range and runs them. |
| **3. Gate and runner** | The gate is the only path to data (DuckDB over parquet). It builds `SELECT group, count(*) AS n, … HAVING n >= 10` itself, so counts cannot be forged, and it refuses identifiers, dates and near-unique values. The runner executes the design and returns only the cohort flow, Table 1 (cells under 10 shown as `<10`), the outcome rate and the model output. |
| **4. Critics** | A **rule critic** checks validity (look-ahead, immortal time, duplicated units, landmarks), feasibility, data quality, model adequacy and protocol traceability. An **LLM clinical critic** from a different model family reads the protocol, design and aggregates, with no tools and no data. Contextual rules may be waived only if the critic quotes the protocol and code verifies the quote. |
| **5. Evaluator** | Optional. Compares the final cohort with your ground-truth CSV after the loop has finished. The agent never sees the ground truth. |

**Models:** only explainable ones: logistic regression (odds ratios), score evaluation (AUROC, calibration), descriptive statistics, and Cox regression (hazard ratios, including time-dependent exposure).

---

## Requirements

- **Python:** tested on 3.14. The dependencies are in `requirements.txt` (DuckDB, pandas, statsmodels, pydantic …). No GPU is needed.
- **Data:** [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/). You need your own PhysioNet credentialed access; no data ships with this repository. Allow about 16 GB of RAM.
- **An LLM endpoint** speaking the OpenAI `/v1/chat/completions` API. The default configuration targets the KHDP gateway used at the datathon; any compatible endpoint works.

---

## Quick start

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt

# 1. data: place MIMIC-IV 3.1 at physionet/mimiciv/3.1/{hosp,icu}/*.csv.gz, then convert it to parquet once
python -m ucd.build_parquet mimic

# 2. check the pipeline on your data with a scripted model (no LLM call); must end with SELF-TEST PASSED
python -m ucd.selftest

# 3. check the LLM: endpoint, models per role, one test call; must end with "JSON ok: True"
python -m ucd.llm check

# 4. run the example
python -m ucd samples/t3_sodium_question
```

The example asks: *"Is low sodium at ICU admission linked to death in adult ICU patients?"*

---

## Running a study

```bash
python -m ucd "Is low sodium at ICU admission linked to death in adult ICU patients?"   # a question
python -m ucd my_protocol.txt                                                           # a protocol file (.txt)
python -m ucd samples/<name>                                                            # a sample folder
python -m ucd                                                                           # asks you for the question
```

| Option | Effect |
|---|---|
| `--auto` | never ask; the agent takes its recommended choice at every question |
| `--gt cohort.csv` | compare the final cohort with your ground truth at the end |
| `--blind` | hide outcome associations from the agent until it freezes the design |
| `--dataset <name>` | use another dataset registered in `ucd/datasets.yaml` |

**A full protocol** can be any free text. One line per item leads to the fewest questions:

```text
Objective: Is dysnatremia at ICU admission associated with in-hospital mortality?
Population: adults at their first ICU stay with a serum sodium within 24 h of ICU admission
Exclusions: none
Time zero: ICU admission
Exposure: first sodium within 24 h, categories < 125, 125-129, 130-134, 135-145 (reference), 146-150, > 150
Outcome: in-hospital death
Covariates: age, sex, admission type
Analysis: logistic regression
Primary estimand: adjusted OR for sodium < 125 vs 135-145
```

If the protocol names a score or criterion the data cannot define by itself, give its components and thresholds. The agent must not invent them.

**Questions** offer at most 4 numbered choices, one marked as recommended (Enter accepts it). `A` lets the agent decide the rest of the current step. At the final protocol check you can open `protocol.md` in your editor and write anything in your own words.

**Stopping early:** type `q` + Enter (checked between agent steps) or press Ctrl+C. A `final.md` is still written from what has run.

**Invalid questions:**
- If intake finds the question cannot be studied as asked, it offers Stop, Rephrase or Continue.
- If a design yields fewer than 100 units, fewer than 20 events or an exposure without variation, the agent goes back to the design, and then asks you how to proceed.

### Output (`runs/ucd/<question>/<time>/`, or `samples/<name>/result/`)

| File | Contents |
|---|---|
| `final.md` | the agent's summary: design, cohort flow, result with CI, iterations, limitations. If no design ever ran, an `AUTO-GENERATED` summary says so |
| `report.md` (samples) | deterministic analysis: each design choice and its reason, undeclared deviations, iteration history, and the ground-truth comparison |
| `protocol.md`, `request.md`, `questions.jsonl` | the agreed protocol, your original words, every question and answer |
| `mimic/iter_<k>/` | per design: `spec.yaml`, `card.json` (aggregates), `critique.json`, `feedback.md` |
| `loop_log.jsonl`, `verdict.json` | the audit trail of every agent step, and the run's one-word status |

`python -m ucd status --run-dir <run>` summarises a run, and `python -m ucd.sample report samples/<name>` rebuilds a report without any LLM call.

---

## Configuration

LLM settings live in [`ucd/config.yaml`](ucd/config.yaml), and environment variables override them.

| Role | Default model | Job |
|---|---|---|
| `planner` | `deepseek-v4-pro-ax` | the design loop (many turns, temperature 0) |
| `intake` | `deepseek-v4-pro-ax` | protocol check and questions |
| `critic` | `qwen3-8-max-ax` | clinical critic and waiver judge (a different model family from the planner) |

| Variable | Effect |
|---|---|
| `UCD_LLM_BASE_URL` | the endpoint (default: the KHDP gateway) |
| `UCD_LLM_MODEL` / `UCD_<ROLE>_MODEL` | one model for every role / for one role |
| `UCD_LLM_API_KEY_FILE` | a file holding the API key (the client never reads keys from `config.yaml`) |
| `UCD_LLM_THINKING=1` | turn on reasoning ("thinking") mode |
| `UCD_CRITIC_BACKEND` | `openai_compat` (default in `python -m ucd`), `claude_cli` (Claude Code CLI) or `off` |
| `UCD_THREADS`, `UCD_MEMORY` | DuckDB limits (default: all cores, 80 % of RAM) |

The agent uses no native tool-calling. Each turn the model replies with one JSON action (`lookup`, `explore`, `read`, `write`, `edit`, `run`, `ask`, `infeasible`, `done` …), and `ucd/loop.py` executes it.

Before trusting the waivers of a new critic model, run `python -m eval.judge_eval`. It measures false waivers and false denials on planted cases.

---

## Privacy model

- **What the LLM sees:** schemas, public documentation, dictionary labels, and aggregates over groups of at least 10 rows. Row-level results stay inside the runner and the evaluator's folders, which the agent is never told about.
- **What the gate refuses:** identifiers (including ids renamed through an alias), dates, near-unique group values, and min/max/list aggregates. Every query is logged and counted against a budget of 60 per run.
- **Residual risks:**
  - Combining many aggregates (differencing) can still reveal individuals.
  - An id transformed by arithmetic is not recognised as an id.
  - Only the query budget and the logs mitigate these; there is no formal differential privacy.
- **Your responsibility:** a hosted LLM still receives schema text and aggregates. Make sure your data use agreement allows this. Treat every run folder and log as derived data (they are git-ignored).

---

## Comparing with a ground truth

```bash
python -m eval.groundtruth --run-dir <run> --gt cohort.csv [--dataset mimic]
```

- **Input:** a CSV with one row per unit, an id column (`stay_id`, `hadm_id` or `subject_id`) and an optional 0/1 outcome column.
- **Comparison:** both cohorts are aligned to a common level (stay → admission → patient).
- **Output:** n, Jaccard, recall, precision and outcome agreement. The sample report also names the flow step where the agent lost ground-truth units.
- **No LLM is involved:** only CSV is accepted, so no model ever reads or interprets the reference.

A sample folder runs the comparison automatically when it holds a `gt.csv`. Such files contain real patient ids, so they are git-ignored.

---

## Adding a dataset

UCD is not tied to MIMIC. A dataset is one entry in [`ucd/datasets.yaml`](ucd/datasets.yaml): label, description, raw source folder, unit levels (id column and linking table), and optional dictionary tables.

1. Write the entry. Optionally add `ucd/docs/<name>_notes.md`, which the planner reads.
2. Run `python -m ucd.build_parquet <name>`.
3. Run `python -m ucd "question" --dataset <name>`.

---

## Insight

An agent that never sees a patient row can still design plausible cohort studies. Its errors, however, are subtle design choices rather than obvious failures, and the guardrails meant to catch them can introduce bias of their own. Automated checks therefore support careful human review; they do not replace it.

---

## Repository layout

| Path | Contents |
|---|---|
| `ucd/` | the agent: `chat.py` (CLI), `intake.py`, `loop.py` (design loop), `gate.py`, `runner.py`, `spec.py`, `critic.py`, `llm_critic.py`, `llm.py` (OpenAI-compatible client), `concepts.py` (tested SQL building blocks) |
| `ucd/planner_prompt.md` | the planner's procedure |
| `ucd/datasets.yaml`, `ucd/docs/` | the dataset registry and the documentation the agent reads |
| `ucd/topics.yaml`, `ucd/packs.yaml` | stored protocol briefs (`--topic`) and topic packs |
| `eval/` | evaluator only, hidden from the agent: ground-truth comparison, reference studies, waiver-judge test, privacy audit |
| `third_party/mimic-code/` | the MIT-licensed mimic-code DuckDB concepts (KDIGO, SOFA, Sepsis-3 …) |
| `samples/t3_sodium_question/` | the example study |
| `docs/critic_rules.md` | rules for event-anchored and landmark designs |

---

## Development

```bash
UCD_CRITIC_BACKEND=off python -m pytest -q        # tests, no LLM calls (the data tests skip without parquet)
python -m ucd card --dataset mimic                # the Data Card the planner sees
python -m ucd lookup --dataset mimic "norepinephrine" --kind items
python -m ucd run --dataset mimic --run-dir R --spec spec.yaml     # one design iteration by hand
python -m ucd.diag summary                        # local error and run log (logs/ucd_events.jsonl)
```

Never commit data, run folders, `gt.csv` or cohort files: `.gitignore` covers them.

## License

MIT, see [`LICENSE`](LICENSE). The licence covers the code only: MIMIC-IV data stays under its PhysioNet data use agreement, and the vendored mimic-code SQL keeps its own MIT licence (`third_party/mimic-code/LICENSE`).

## Acknowledgements

MIMIC-IV (PhysioNet, MIT Laboratory for Computational Physiology). The concept SQL comes from [mimic-code](https://github.com/MIT-LCP/mimic-code) (MIT License). Built at the Korea Clinical Datathon 2026.
