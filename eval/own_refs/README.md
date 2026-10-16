# Own evaluator references: t3, t5, t6

These are **our own hand-written reference cohorts**, not published code. mimic-code has no cohort for these
topics. They were written from `ucd/topics.yaml` and `eval/papers.yaml` only, without looking at any agent run.
They serve as the `E_code` comparison in `eval/evaluate.py`.

```
python eval/own_refs/build_own_refs.py   # -> eval/mimiccode/ref/t{3,5,6}_*.parquet + own_refs_summary.json
```

## Common definitions
- **Age** = `anchor_age + year(intime) - anchor_year`, with adults defined as age >= 18. Outcome = `admissions.hospital_expire_flag`.
- **Surgical** = any surgical service (CSURG, NSURG, ORTHO, PSURG, SURG, TRAUM, TSURG, VSURG, GU, GYN, ENT) in `hosp.services`
  up to ICU intime + 24 h. **Emergency** = admission_type EW EMER., DIRECT EMER. or URGENT.
- **ICU stay** = a row of `icu.icustays`. This follows MIMIC-IV, which also includes Neuro Intermediate and Neuro Stepdown.

## t3_dysnatremia
- **Unit:** each patient's first ICU stay (adult).
- **Exclusions:** no lab-chemistry sodium (itemid 50983), or missing hospital outcome.
- **Exposure:** `na_first_24h` = the earliest 50983 value in [intime, intime + 24 h]. Blood-gas sodium is not used.
- **Categories:** <125, 125–<130, 130–<135, 135–145 (reference), >145–150, >150–155, >155.
- **Model:** logistic regression with age, sex, surgical, emergency admission and mimic-code first-day SOFA (missing set to 0) as covariates. SOFA stands in for SAPS II.

## t5_readmission
- **Index:** the first ICU stay of each hospitalization (adult).
- **Exclusions:**
  - Death in the ICU: deathtime <= ICU outtime + 1 h.
  - Not discharged to a ward. The next `hosp.transfers` event at or after ICU outtime must be a transfer to a careunit that is not an ICU, an ED, the Discharge Lounge or UNKNOWN. ICU-to-discharge and ICU-to-Discharge-Lounge are therefore excluded.
- **Outcome:** `readmit` = any later `icustays` row in the same hospitalization. Because the index stay went to a ward, a ward stay always lies in between. ICU-to-ICU moves are merged into one stay by `icustays`.
- **`first_icu_stay_patient`:** True only for each patient's first eligible hospitalization. The evaluator restricts to these rows.

## t6_transfusion
- **Unit:** each patient's first ICU stay (adult).
- **Exclusion:** routine post-operative monitoring, operationalised as ELECTIVE or SURGICAL SAME DAY admission + surgical service + ICU LOS < 24 h.
- **Exposure:** `rbc_any` = `inputevents` itemid 225168 ("Packed Red Blood Cells") with amount > 0 and starttime within [intime, outtime].
  - OR and PACU RBC intake (226368, 227070) are not counted.
  - `first_rbc_hours` = hours from ICU intime to the first RBC.
- **Trigger:** `trigger_hb` = the lowest Hb (51222 or 50811, 0–25 g/dL) on the calendar day of the first RBC. This includes values measured after the transfusion that day, as in ICON.
- **Cox model:**
  - Origin ICU intime; follow-up ends at hospital death or discharge (censored).
  - Counting-process split at the first RBC: an unexposed interval [0, t_rbc) and an exposed interval [t_rbc, end].
  - Adjusted for age, sex and admission Hb (the first Hb in [intime - 6 h, intime + 24 h]; complete cases).
  - Uses statsmodels PHReg with entry times and Breslow ties.
  - No severity or indication adjustment, so the estimate is confounded by indication.

## Evaluator caveats
- **t3 Spearman agreement:** agent stays files carry sodium category labels, not raw sodium. The number extracted from such a label is not monotone in sodium, so the Spearman agreement for t3 is unreliable.
- **t6 Spearman agreement:** agent files carry RBC flags, not `trigger_hb`, so the t6 agreement is weak by construction.
- **t5 outcome:** the reference outcome rate is `readmit`, not mortality.
