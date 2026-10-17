# Final Design Summary: Low Sodium at ICU Admission and In-Hospital Death

## Research Question
Is low sodium at ICU admission associated with in-hospital death in adult ICU patients?

## Final Design

| Component | Specification |
|-----------|---------------|
| Population | Adult patients (age ≥ 18) with a first ICU stay in MIMIC-IV 3.1 |
| Time zero | ICU admission time (icu.intime) |
| Exposure | First serum sodium value recorded in the ICU within 24 hours of ICU admission, below 135 mmol/L (vs. ≥ 135 mmol/L) |
| Outcome | In-hospital death (hospital_expire_flag = 1) |
| Covariates | Age, sex, admission type (collapsed to 4 groups) |
| Model | Logistic regression (odds ratios) |
| Primary estimand | Odds ratio for in-hospital death comparing low sodium (< 135 mmol/L) to normal sodium (≥ 135 mmol/L), adjusted for age, sex, and admission type |

## Results

### Cohort Flow
| Step | Criterion | Units Remaining |
|------|-----------|-----------------|
| Start | First ICU stays | 65,366 |
| Inclusion | Age ≥ 18 at ICU admission | 65,366 |
| Inclusion | Has sodium measured within 24h of ICU admission | 63,402 |

### Outcome
- In-hospital death: 6,592 / 63,402 = 10.4% (95% CI [10.2%, 10.6%])

### Primary Estimate
- **Adjusted odds ratio for low sodium: 1.452 (95% CI [1.367, 1.542], p < 0.001)**

### Sensitivity Analysis (24-hour landmark)
- Excluding deaths within the first 24 hours: OR = 1.463 (95% CI [1.372, 1.561], p < 0.001)

## Iteration History

| Iteration | Changes | Reason |
|-----------|---------|--------|
| 1 | Initial design | Established cohort, exposure, outcome, covariates |
| 1 → 2 | Fixed sodium window filter; added low_sodium to predictors; collapsed admission_type; added 24h landmark sensitivity | Fixed SQL bug (pre-ICU sodium rows); fixed model specification; resolved quasi-complete separation; addressed immortal-time bias |
| 2 → 3 | Corrected limitations text; attempted V3 waiver | Fixed documentation error; waiver refused by judge |

## Limitations

1. **Immortal-time bias**: 14.6% of deaths occur within the 24-hour exposure window. The protocol fixes both the exposure window and outcome from time zero, so this is inherent. A 24-hour landmark sensitivity analysis (OR = 1.463) shows the association is robust.

2. **Exclusion of patients without sodium**: 3.0% of first ICU stays were excluded for lacking a sodium measurement within 24 hours of ICU admission.

3. **Admission type heterogeneity**: The URGENT category captures a heterogeneous group including inter-facility transfers, which may explain its higher mortality rate (13.5%) compared to EMERGENCY (12.1%).

4. **Sodium measurement pooling**: Both chemistry (itemid 50983) and blood-gas (itemid 50824) sodium values were pooled, per dataset guidance, as the protocol did not specify which to use.

## Assumptions Made

- **First ICU stay per patient**: The protocol specified "adult ICU patients" without qualification; we used first ICU stay per patient (not per hospitalization).
- **Sodium item pooling**: Chemistry and blood-gas sodium items were pooled as the protocol did not specify a source.
- **Admission type collapsing**: Collapsed from 9 to 4 categories to avoid quasi-complete separation in the logistic model.
- **Outcome timing**: In-hospital death was defined as hospital_expire_flag = 1, regardless of whether death occurred during the ICU stay or later in the hospitalization.