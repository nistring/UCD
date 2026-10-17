Objective: To assess whether low sodium at ICU admission is associated with in-hospital death in adult ICU patients.
Population: Adult patients (anchor_age >= 18) with an ICU stay in MIMIC-IV 3.1.
Exclusions: Exclude patients with no sodium measured within 24h of ICU admission.
Time zero: ICU admission time (icu.intime)
Exposure: First serum sodium value recorded in the ICU within 24 hours of ICU admission, below 135 mmol/L
Outcome: In-hospital death (hospital_expire_flag = 1)
Covariates: Age, sex, and admission type
Analysis: Logistic regression (odds ratios)
Primary estimand: Odds ratio for in-hospital death comparing patients with first ICU sodium < 135 mmol/L to those with first ICU sodium >= 135 mmol/L, adjusted for age, sex, and admission type.

Clarifications at intake:
- How should low sodium at ICU admission be defined? -> First serum sodium value recorded in the ICU within 24 hours of ICU admission, below 135 mmol/L (auto: the agent's recommended choice, not the user's)
- Which death outcome should be used? -> In-hospital death (hospital_expire_flag = 1) (auto: the agent's recommended choice, not the user's)
- What should be the time zero for the analysis? -> ICU admission time (icu.intime) (auto: the agent's recommended choice, not the user's)
- Which patients should be excluded? -> Exclude patients with no sodium measured within 24h of ICU admission (auto: the agent's recommended choice, not the user's)
- Which covariates should be adjusted for? -> Age, sex, and admission type (auto: the agent's recommended choice, not the user's)
- Which analysis method should be used? -> Logistic regression (odds ratios) (auto: the agent's recommended choice, not the user's)
