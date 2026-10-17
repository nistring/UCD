# UCD: a cohort-design agent for studying agentic AI on clinical data

UCD is an LLM agent that turns a clinical research question into an observational cohort study on an EHR database. It writes the protocol with you, designs the cohort in SQL, runs an explainable model, and revises the design under two critics. **The language model never sees a patient row**: it works only from schemas, documentation and aggregate counts.

UCD was built for the Korea Clinical Datathon 2026 (theme *"Agentic AI: Light & Shadow"*). Its purpose is to observe how an AI agent behaves when it works on clinical data: what it gets right, where it fails, and which safeguards help.

> [!WARNING]
> **Not for clinical use.** This is a research tool for studying how agentic AI behaves when it processes clinical data. It is not a medical device. Its output has not been validated and must not inform patient care, clinical decisions or published findings unless a qualified researcher has verified it independently. The agent can and does make design errors: observing them is the point of the project.

## Contents

1. [Overview](#1-overview)
2. [Quick start](#2-quick-start)
3. [Usage](#3-usage)
4. [How it works](#4-how-it-works)
5. [Experiments: baselines, ablations, models and studies](#5-experiments)
6. [Configuration](#6-configuration)
7. [Privacy model](#7-privacy-model)
8. [Datasets](#8-datasets)
9. [Ground-truth comparison](#9-ground-truth-comparison)
10. [Repository layout](#10-repository-layout)
11. [Development](#11-development)
12. [Insight](#12-insight)
13. [License](#13-license)

---

## 1. Overview

| | |
|---|---|
| **Input** | a research question or a full protocol, as free text |
| **Output** | the agreed protocol, the cohort design (spec + SQL), cohort flow, Table 1, an explainable model, and a summary with limitations |
| **Data** | MIMIC-IV 3.1 is configured. Other datasets are added in a registry; the prompts and rules name no dataset |
| **Models** | any OpenAI-compatible chat endpoint, one model per role (intake, planner, critic) |
| **Analyses** | logistic regression (odds ratios), score evaluation (AUROC, calibration), descriptive statistics, Cox regression (hazard ratios, time-dependent exposure), propensity weighting |
| **Research use** | built-in baselines and ablations that compare the full agent with simpler setups |

## 2. Quick start

**Requirements**
- Python 3 (tested on 3.14), with the packages in `requirements.txt`. No GPU is needed; allow about 16 GB of RAM.
- [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/), through your own PhysioNet credentialed access. No data ships with this repository.
- An OpenAI-compatible LLM endpoint (see [Configuration](#6-configuration)).

```bash
pip install -r requirements.txt

# 1. place MIMIC-IV 3.1 at physionet/mimiciv/3.1/{hosp,icu}/*.csv.gz, then convert it to parquet once
python -m ucd.build_parquet mimic

# 2. check the pipeline on your data with a scripted model (no LLM call); ends with SELF-TEST PASSED
python -m ucd.selftest

# 3. check the LLM endpoint and the model of each role; ends with "JSON ok: True"
python -m ucd.llm check

# 4. run the example study
python -m ucd samples/example/t3_sodium_question
```

The example asks: *"Is low sodium at ICU admission linked to death in adult ICU patients?"*

## 3. Usage

### Starting a study

```bash
python -m ucd "Is low sodium at ICU admission linked to death in adult ICU patients?"   # a question
python -m ucd my_protocol.txt                                                           # a protocol file (.txt)
python -m ucd samples/<name>                                                            # a sample folder
python -m ucd                                                                           # asks for the question
```

| Option | Effect |
|---|---|
| `--auto` | never ask; the agent takes its recommended choice at every question |
| `--gt cohort.csv` | compare the final cohort with your ground truth at the end |
| `--blind` | hide outcome associations until the agent freezes its design |
| `--dataset <name>` | another dataset from `ucd/datasets.yaml` |

### Writing a protocol

Any free text works. One line per item leads to the fewest questions:

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

### During a run

- **Questions:**
  - Each question offers up to 4 numbered choices, one of them recommended; Enter accepts it.
  - `A` lets the agent decide the rest of the current step.
  - At the final protocol check you can edit `protocol.md` in your own words.
- **Stopping early:** `q` + Enter, or Ctrl+C. A summary of what has run is still written.
- **Questions that cannot be studied:**
  - Intake offers Stop, Rephrase or Continue.
  - A design with fewer than 100 units, fewer than 20 events or an exposure without variation goes back to the design loop, and then you decide how to proceed.

### Output

Runs are written to `runs/ucd/<question>/<time>/`, or to `samples/<name>/result/` for a sample.

| File | Contents |
|---|---|
| `final.md` | the summary: design, cohort flow, result with CI, iterations, limitations (marked `AUTO-GENERATED` if the agent did not write it) |
| `report.md` | samples only: every design choice with its reason, undeclared deviations, the iteration history, and the ground-truth comparison |
| `protocol.md`, `request.md`, `questions.jsonl` | the agreed protocol, your original words, every question and answer |
| `<dataset>/iter_<k>/` | per design: `spec.yaml`, `card.json` (aggregates), `critique.json`, `feedback.md` |
| `loop_log.jsonl`, `verdict.json` | every agent step, and the run's one-word status |

`python -m ucd status --run-dir <run>` summarises a run; `python -m ucd.sample report samples/<name>` rebuilds a report without any LLM call.

## 4. How it works

```
 question / protocol (.txt)
        │
        ▼
 [1 Intake LLM]   checks it against a standard protocol format and the dataset schema,
        │         asks about gaps; you confirm or edit protocol.md
        ▼
 [2 Planner LLM] ── design/spec.yaml + design/query.sql ──► [3 Gate + Runner]  (deterministic, no LLM)
        ▲                                                         │ aggregates only
        │                                                         ▼
        └──── feedback ◄──── [4 Rule critic + LLM clinical critic]
                  (at most 3 executed designs; stops when no blocking finding remains)
        │
        ▼
 final.md ──► [5 Evaluator]  optional ground-truth CSV, compared deterministically, never fed back
```

| Component | Role |
|---|---|
| **1. Intake** | Completes the protocol: objective, population, exclusions, time zero, exposure, outcome, covariates, analysis and primary estimand. It sees the schema, so it offers only measurable choices. The confirmed protocol is the gold standard. |
| **2. Planner** | Looks up codes and explores the data through aggregate queries, then writes a design spec and its SQL, and edits them by line range between runs. |
| **3. Gate and runner** | The only path to data (DuckDB over parquet). The gate computes `count(*) … HAVING n >= 10` itself and refuses identifiers, dates and near-unique values. The runner returns only aggregates: flow, Table 1, outcome and model output. |
| **4. Critics** | Rule critic: deterministic, topic-independent checks of validity (look-ahead, immortal time, duplicated units, landmarks), feasibility, data quality, model adequacy and protocol traceability. LLM clinical critic: a different model family that reads the protocol, design and aggregates, with no tools and no data. A contextual rule can be waived only if the critic quotes the protocol and code verifies the quote. |
| **5. Evaluator** | Compares the final cohort with a ground truth after the loop has finished. The agent never sees it. |

**What the prompts know.** The prompts and critic rules are dataset-generic: they speak in the agent's own columns (`unit_id`, `index_time`, `unit_end_time`). Everything specific to a dataset (table and column names, how death is recorded, when codes are dated) comes from the Data Card: the registry, the schema, and `ucd/docs/<dataset>_notes.md`.

## 5. Experiments

`eval/experiments.py` runs sample studies under several agent configurations ("arms") and compares them.

```bash
python -m eval.experiments list                                              # the arms
python -m eval.experiments run samples/example/t3_sodium_question --arms baselines   # or ablations, all, or a,b,c
python -m eval.experiments run samples/tasks/*/* --arms all --out runs/experiments/tasks         # the rejection test
python -m eval.experiments summary runs/experiments/tasks                    # rebuild the summary table
```

### Samples

| Folder | Contents |
|---|---|
| `samples/example/t3_sodium_question/` | the example study (a one-line question), with a published example run in `result/` |
| `samples/tasks/answerable/T0xx/` | 20 artificially generated cohort tasks with a valid answer: the agent should build the cohort (verdict `normal`) |
| `samples/tasks/not_answerable/T0xx/` | 4 tasks the agent should refuse or qualify: a population that cannot exist in the data, a value the data caps, criteria that contradict each other, a follow-up longer than the data covers |
| `samples/tasks/pairs/<change>/T0xx/` | invalid twins of answerable tasks, one detail changed (see the minimal-pair study below) |
| `samples/pressure/<base>/{neutral,pressure}/` | protocols with and without a sponsor's expected result (see the pressure study below) |

Each task's `sample.yaml` lists the verdicts it accepts. A task can also hold a `gt.csv` with the expected cohort; it holds real ids, so it is never published. `python -m ucd.sample score samples/tasks/*/*` reports the share of accepted verdicts, the cohort agreement, and how many invalid tasks the agent rejected.

### Arms

**Baselines** use the same model and data access with less structure around it:

| Arm | Setup |
|---|---|
| `full` | UCD as designed: intake, planner with tools, rule critic, LLM critic |
| `single_generator` | one LLM call: request and Data Card in, a design out; run once, no feedback |
| `single_agent` | one LLM agent with tools (lookup, explore, run), no intake and no critics; it sees only execution results |

**Ablations** switch off one part of the full system:

| Arm | What is removed |
|---|---|
| `no_intake` | the raw request goes straight to the planner |
| `no_llm_critic` / `no_rule_feedback` / `no_critics` | the LLM critic, the rule findings, or both |
| `no_waivers` | waiver requests are ignored |
| `no_explore` | lookup and explore (design from the Data Card only) |
| `no_concepts` | the tested concept library and topic packs |
| `no_notes` | the dataset notes (schema only) |
| `one_design` | iteration: one executed design |
| `blinded` | (added rather than removed) outcome-blinded design phase |
| `same_model_critic` | model diversity: the critic uses the planner's model |

Each arm is a set of `run_config.json` switches, so the same switches work on any run: `llm_critic`, `rule_feedback`, `waivers`, `explore`, `concepts`, `notes`, `max_iter`, `blind_outcome`.

### Comparing models

```bash
python -m eval.experiments run <samples> --arms no_intake --jobs 4 \
    --planner deepseek-v4-pro-ax --planner deepseek-v4-pro-ax:think --planner qwen3-8-max-ax   # planner models
python -m eval.experiments intake <samples> --models deepseek-v4-pro-ax,qwen3-8-max-ax --out <dir>   # intake models
UCD_CRITIC_MODEL=<model> python -m eval.judge_eval --reps 2                                        # critic models
```

`:think` turns on reasoning for the planner only. The intake check asks whether each model stops at the invalid tasks and passes the valid ones; the judge test counts false waivers and false denials on planted cases.

### Results

- Every arm with intake starts from the same protocol per sample, written once in auto mode.
- The rule critic scores every executed design in every arm. Where an arm hides it from the planner, it is stored on the evaluator side, so `I_rule` stays comparable.
- `<out>/results.csv` has one row per run: designs run, turns, invalid turns, the longest streak of edits that changed nothing, seconds, cohort size, primary estimate, `I_rule`, open rule blocks, the verdict and whether the task accepts it, and Jaccard / recall / precision when the sample has a `gt.csv`.
- `<out>/summary.md` holds the means per arm and planner, the share of accepted verdicts, and the number of invalid tasks rejected.

### Studies

| Study | Question | Commands |
|---|---|---|
| **Minimal pairs** | Does the agent notice an invalid detail, or does it refuse everything? Each answerable task gets twins that differ in one detail: children in an adults-only dataset, an age above the data's cap, an impossible earlier ICU stay, a follow-up longer than the data covers, an unrecorded variable. | `python -m eval.minimal_pairs samples/tasks/answerable/T0xx ...`, then run the originals and `samples/tasks/pairs/*/*` |
| **Rule attribution** | Which critic findings move the cohort toward the ground truth, and which push it away? Each revision's change in Jaccard is credited to the findings it resolved (associative, not causal). | `python -m eval.rule_attribution runs/experiments/<out>` |
| **Outcome pressure** | Does a sponsor's expected result change the design, the estimate or the claims? Two bases (a real association and a null by construction), each neutral and pressured, run with and without outcome blinding. | `python -m eval.experiments run samples/pressure/*/* --arms full,blinded`, then `python -m eval.pressure report runs/experiments/<out>` |

## 6. Configuration

LLM settings live in [`ucd/config.yaml`](ucd/config.yaml); environment variables override them.

| Role | Recommended model and settings | Why (observed in this project's runs, not a benchmark) |
|---|---|---|
| `planner` | `deepseek-v4-pro-ax`, thinking off, temperature 0 | ran a design by turn 8 in its first live run; the example study matched the ground truth (Jaccard 0.994). A smaller model (`qwen3-8-flash-next-ax`) spent 24 of 30 turns on lookups and never ran a design |
| `intake` | `deepseek-v4-pro-ax` | asked three sensible questions in 18 s with the schema in view |
| `critic` | `qwen3-8-max-ax` | a different model family from the planner, so the review is independent; 24 s per review, and it caught a wrong sodium window, an undeclared landmark and model separation. `glm-5-3-ax` returned no valid reply after 259 s |

Reasoning ("thinking") mode stays off by default: it is slower, and long replies were cut off at the token limit before. It is worth testing as a planner variant (`--planner deepseek-v4-pro-ax:think`, see [Comparing models](#comparing-models)). The models above are those of the KHDP gateway; any OpenAI-compatible endpoint works.

| Variable | Effect |
|---|---|
| `UCD_LLM_BASE_URL` | the endpoint (default: the KHDP gateway used at the datathon) |
| `UCD_LLM_MODEL` / `UCD_<ROLE>_MODEL` | one model for every role, or for one role |
| `UCD_LLM_API_KEY_FILE` | a file holding the API key (the client never reads keys from `config.yaml`) |
| `UCD_LLM_THINKING=1` | turn on reasoning mode |
| `UCD_CRITIC_BACKEND` | `openai_compat` (the configured endpoint; default in `python -m ucd` and the experiments), `claude_cli` (Claude Code CLI) or `off` |
| `UCD_THREADS`, `UCD_MEMORY` | DuckDB limits (default: all cores, 80 % of RAM) |

The agent uses no native tool-calling: each turn the model replies with one JSON action, and `ucd/loop.py` executes it. Before trusting the waivers of a new critic model, run `python -m eval.judge_eval`; it counts false waivers and false denials on planted cases.

## 7. Privacy model

- **What the LLM sees:** schemas, public documentation, dictionary labels, and aggregates over groups of at least 10 rows. Row-level results stay inside the runner and the evaluator's folders, which the agent is never told about.
- **What the gate refuses:** identifiers (also under an alias), dates, near-unique group values, and min/max/list aggregates. Every query is logged and counted against a budget of 60 per run.
- **Residual risks:**
  - Combining many aggregates (differencing) can still reveal individuals.
  - An id transformed by arithmetic is not recognised.
  - Only the budget and the logs mitigate these; there is no formal differential privacy.
- **Your responsibility:** a hosted LLM still receives schema text and aggregates. Make sure your data use agreement allows this. Treat every run folder and log as derived data (they are git-ignored).

## 8. Datasets

A dataset is one entry in [`ucd/datasets.yaml`](ucd/datasets.yaml):

| Field | Meaning |
|---|---|
| `label`, `description` | shown to the models (public documentation only) |
| `source`, `parquet_env` | where the raw tables and the parquet copy live |
| `units` | unit levels of analysis, fine to coarse (id column and linking table); they are the allowed values of `unit_of_analysis` |
| `death` | which columns record in-hospital deaths only, and which also cover deaths after discharge |
| `lookups`, `concepts` | dictionary tables for code lookup, and an optional library of tested SQL concepts |

To add a dataset:
1. Write its entry.
2. Optionally write `ucd/docs/<name>_notes.md`, which the planner and the critic read.
3. Run `python -m ucd.build_parquet <name>`.
4. Run `python -m ucd "question" --dataset <name>`.

## 9. Ground-truth comparison

```bash
python -m eval.groundtruth --run-dir <run> --gt cohort.csv [--dataset mimic]
```

- **Input:** a CSV with one row per unit, an id column (e.g. `stay_id`, `hadm_id`, `subject_id`) and an optional 0/1 outcome column.
- **Comparison:** both cohorts are aligned to a common unit level.
- **Output:** n, Jaccard, recall, precision and outcome agreement.
- **No LLM involved:** only CSV is accepted, so no model ever reads or interprets the reference.

A sample folder with a `gt.csv` runs the comparison automatically. These files hold real patient ids, so they are git-ignored.

## 10. Repository layout

| Path | Contents |
|---|---|
| `ucd/` | the agent: `chat.py` (CLI), `intake.py`, `loop.py`, `gate.py`, `runner.py`, `spec.py`, `critic.py`, `llm_critic.py`, `llm.py`, `concepts.py` |
| `ucd/planner_prompt.md` | the planner's procedure |
| `ucd/datasets.yaml`, `ucd/docs/` | the dataset registry and the documentation the agent reads |
| `ucd/topics.yaml`, `ucd/packs.yaml` | stored protocol briefs (`--topic`) and topic packs |
| `eval/` | evaluator side, hidden from the agent: `experiments.py` (baselines and ablations), `groundtruth.py`, reference studies, waiver-judge test, privacy audit |
| `third_party/mimic-code/` | the MIT-licensed mimic-code DuckDB concepts (KDIGO, SOFA, Sepsis-3 …) |
| `samples/` | the example study and the 24 task cases (see [Samples](#samples)) |
| `docs/critic_rules.md` | rules for event-anchored and landmark designs |
| `tests/` | `python -m pytest` |

## 11. Development

```bash
UCD_CRITIC_BACKEND=off python -m pytest -q        # tests, no LLM calls (data tests skip without parquet)
python -m ucd card --dataset mimic                # the Data Card the planner sees
python -m ucd lookup --dataset mimic "norepinephrine" --kind items
python -m ucd run --dataset mimic --run-dir R --spec spec.yaml     # one design iteration by hand
python -m ucd.diag summary                        # local error and run log (logs/ucd_events.jsonl)
```

Never commit data, run folders, `gt.csv` or cohort files: `.gitignore` covers them.

## 12. Insight

An agent that never sees a patient row can still design plausible cohort studies. Its errors, however, are subtle design choices rather than obvious failures, and the guardrails meant to catch them can introduce bias of their own. Automated checks therefore support careful human review; they do not replace it.

## 13. License

MIT, see [`LICENSE`](LICENSE). The licence covers the code only:
- MIMIC-IV data stays under its PhysioNet data use agreement.
- The vendored mimic-code SQL keeps its own MIT licence (`third_party/mimic-code/LICENSE`).

MIMIC-IV is provided by PhysioNet and the MIT Laboratory for Computational Physiology. UCD was built at the Korea Clinical Datathon 2026.
