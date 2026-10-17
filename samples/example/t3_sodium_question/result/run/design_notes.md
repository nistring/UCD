# Design Notes

## PICO
- **Population:** Adult ICU patients (age ≥ 18) in MIMIC-IV 3.1, first ICU stay per patient.
- **Exposure:** First serum sodium < 135 mmol/L within 24 h of ICU admission.
- **Comparator:** First serum sodium ≥ 135 mmol/L within 24 h of ICU admission.
- **Outcome:** In-hospital death (hospital_expire_flag = 1).

## Assumptions
- "First serum sodium" = earliest charttime within [ICU intime, intime + 24 h], using `concept.chemistry` (sodium from chemistry labs). Blood-gas sodium is not used because the protocol says "serum sodium".
- Patients with no sodium in the window are excluded (per protocol).
- Age at admission computed as `anchor_age + (year(admittime) - anchor_year)`.
- Admission type from `hosp.admissions.admission_type`.

## Bias threats to guard against
- **Immortal time:** Exposure window is 24 h; outcome counted from time zero (ICU admission), so early deaths before sodium measurement are possible. This is inherent to the protocol's "first value within 24 h" definition and will be noted.
- **Look-ahead:** Sodium values after 24 h must not be used.
- **Confounding:** Adjusted for age, sex, admission type as specified.