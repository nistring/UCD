# Critic rules for event-time-zero / landmark designs (T1-T13)

Origin: written for an event-anchored protocol (2026-10-17); its disease-specific rules (T5, T12) were removed in 0.13.2 and the rest is topic-independent. Implemented in 0.10.0: T2, T3, T6, T7, T10, T11, T13 and the finding order (`ucd/critic.py`, `critic.event_rules`). Not implemented: T1 (index inside unit; the runner's timing checks cover events before index), T4 (look-back bounds would need SQL parsing), T8 (fallback status column) and T9 (covered by P1: every protocol item must be mapped).

Topic-independent; triggered by spec fields, not by protocol text. Tier 1 = mechanical (never waivable), tier 2 = contextual.

| Rule | Tier | Check (deterministic, on spec + `analysis_sql` + aggregates) | Finding |
|---|---|---|---|
| **T1 index inside unit** | 1 | `index_time` must satisfy `unit_start <= index_time <= unit_end_time` for every candidate row (runner computes the violation count from the result table) | BLOCK if > 0 rows |
| **T2 landmark flag present** | 1 | if `outcome_from_h = prediction_time_h = L > 0`, a population flag must exist whose SQL references `index_time + INTERVAL L HOUR` against `unit_end_time` and the death time (regex on the step SQL, plus a runner check that no kept unit has `outcome_time_h <= L`) | BLOCK when missing or when kept units have events before L |
| **T3 exposure before landmark** | 1 | when `exposure_time_h` exists and a landmark L is declared, every exposed kept unit must have `0 <= exposure_time_h <= L` (inclusive at 0 only if the spec says `exposure_inclusive_at_t0: true`) | BLOCK if violated |
| **T4 look-back only** | 2 | each exclusion step whose SQL reads labevents / chartevents / bg must bound `charttime <= index_time` (or `< index_time`); a step with `charttime > index_time` or no upper bound is look-ahead | BLOCK unless waived with a protocol quote |
| **T6 unit level** | 1 | if any `population` item says "per patient", the candidate CTE must partition by `subject_id`, not `hadm_id`; the runner checks `COUNT(DISTINCT subject_key) = COUNT(*)` on candidates | BLOCK |
| **T7 outcome source** | 2 | an outcome item saying "all-cause" / "any death record" with a fixed horizon must use `dod` (or both `deathtime` and `dod`); `hospital_expire_flag` alone is flagged | WARN → BLOCK on 2nd iteration |
| **T8 fallback status kept** | 2 | an exclusion item with a fallback window must produce a status column with values {primary, fallback, unresolved}; "unresolved -> exclude" must be a separate flag so the flow shows it | WARN |
| **T9 flow traceability** | 1 | `protocol_mapping` must map every `exclusion` item to exactly one population flag, in protocol order | BLOCK |
| **T10 flow uninformative** | 1 | the final SELECT of `analysis_sql` must not filter on flag columns (`WHERE incl_…`/`excl_…`), and at least one flag must exclude ≥ 1 unit; a flow in which every step has the same n with ≥ 2 flags is uninformative (observed: 1,518 at all five steps) | BLOCK, fix text: "return candidate rows; the runner applies the flags" |
| **T11 dropped design terms** | 1 | the intake I2 vocabulary check, re-run on `protocol_items.yaml` against `request.md`: landmark, fallback, unresolved, first ICU stay per patient, all-cause, weighting terms and every number-with-unit must each map to an item or a declared deviation | BLOCK (the planner cannot "fix" this; the run returns to intake / the user) |
| **T13 dictionary-label matching** | 2 | exposure, exclusion or population flags built with `ILIKE '%term%'` on `d_items.label` / `d_labitems.label` when a `concept.*` table or an explicit itemid list exists for the concept (e.g. ventilation, vasopressors, blood gas) | WARN → BLOCK on 2nd iteration; fix names the derived table |

**Ranking in the feedback.** Findings about the cohort definition (T1–T13, V-rules on population flags, flow) are listed before
covariate and model findings, and the planner is told to fix them first. In the observed run five covariate BLOCKs preceded the
single WARN about a missing component of the cohort definition; three iterations were spent on covariates while the cohort
stayed at 14 % of the reference size.

Feedback text for the planner names the rule, the offending step (`-- step: <name>`) and the minimal fix. All checks use only the spec, the SQL text and aggregate violation counts; no row is shown.

## Evaluator additions (`eval/groundtruth.py`)
When `gt.csv` has `t0` and/or `exposure` columns besides the id and outcome:
- **T0 agreement** on the overlapping units: share within 1 h, within 6 h, median absolute difference in hours.
- **Exposure agreement**: confusion table and Cohen's kappa on overlapping units.
- **Flow comparison**: the agent's `n_remaining` per flag vs the reference `verification_targets` from `papers.yaml` (step order may differ; match by flag kind).
Printed and saved in `eval/gt/`, never in the run folder.
