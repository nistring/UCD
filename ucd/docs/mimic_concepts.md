# Tested concepts for mimic (use them as tables: concept.<name>)

Built on first use from public, tested SQL; reuse them instead of re-deriving the same definitions.

## comorbidity
- `concept.charlson`: subject_id, hadm_id, age_score, myocardial_infarct, congestive_heart_failure, peripheral_vascular_disease, cerebrovascular_disease, dementia, chronic_pulmonary_disease, rheumatic_disease, peptic_ulcer_disease, mild_liver_disease, diabetes_without_cc, diabetes_with_cc, paraplegia, renal_disease, malignant_cancer, severe_liver_disease, metastatic_solid_tumor, aids, charlson_comorbidity_index

## demographics
- `concept.age`: subject_id, hadm_id, admittime, anchor_age, anchor_year, age
- `concept.icustay_detail`: subject_id, hadm_id, stay_id, gender, dod, admittime, dischtime, los_hospital, admission_age, race, hospital_expire_flag, hospstay_seq, first_hosp_stay, icu_intime, icu_outtime, los_icu, icustay_seq, first_icu_stay
- `concept.icustay_hourly`: stay_id, hr, endtime
- `concept.icustay_times`: subject_id, hadm_id, stay_id, intime_hr, outtime_hr
- `concept.weight_durations`: stay_id, starttime, endtime, weight, weight_type

## firstday
- `concept.first_day_bg`: subject_id, stay_id, lactate_min, lactate_max, ph_min, ph_max, so2_min, so2_max, po2_min, po2_max, pco2_min, pco2_max, aado2_min, aado2_max, aado2_calc_min, aado2_calc_max, pao2fio2ratio_min, pao2fio2ratio_max, baseexcess_min, baseexcess_max, bicarbonate_min, bicarbonate_max, totalco2_min, totalco2_max, hematocrit_min, hematocrit_max, hemoglobin_min, hemoglobin_max, carboxyhemoglobin_min, carboxyhemoglobin_max, methemoglobin_min, methemoglobin_max, temperature_min, temperature_max, chloride_min, chloride_max, calcium_min, calcium_max, glucose_min, glucose_max, potassium_min, potassium_max, sodium_min, sodium_max
- `concept.first_day_bg_art`: subject_id, stay_id, lactate_min, lactate_max, ph_min, ph_max, so2_min, so2_max, po2_min, po2_max, pco2_min, pco2_max, aado2_min, aado2_max, aado2_calc_min, aado2_calc_max, pao2fio2ratio_min, pao2fio2ratio_max, baseexcess_min, baseexcess_max, bicarbonate_min, bicarbonate_max, totalco2_min, totalco2_max, hematocrit_min, hematocrit_max, hemoglobin_min, hemoglobin_max, carboxyhemoglobin_min, carboxyhemoglobin_max, methemoglobin_min, methemoglobin_max, temperature_min, temperature_max, chloride_min, chloride_max, calcium_min, calcium_max, glucose_min, glucose_max, potassium_min, potassium_max, sodium_min, sodium_max
- `concept.first_day_gcs`: subject_id, stay_id, gcs_min, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable
- `concept.first_day_height`: subject_id, stay_id, height
- `concept.first_day_lab`: subject_id, stay_id, hematocrit_min, hematocrit_max, hemoglobin_min, hemoglobin_max, platelets_min, platelets_max, wbc_min, wbc_max, albumin_min, albumin_max, globulin_min, globulin_max, total_protein_min, total_protein_max, aniongap_min, aniongap_max, bicarbonate_min, bicarbonate_max, bun_min, bun_max, calcium_min, calcium_max, chloride_min, chloride_max, creatinine_min, creatinine_max, glucose_min, glucose_max, sodium_min, sodium_max, potassium_min, potassium_max, abs_basophils_min, abs_basophils_max, abs_eosinophils_min, abs_eosinophils_max, abs_lymphocytes_min, abs_lymphocytes_max, abs_monocytes_min, abs_monocytes_max, abs_neutrophils_min, abs_neutrophils_max, atyps_min, atyps_max, bands_min, bands_max, imm_granulocytes_min, imm_granulocytes_max, metas_min, metas_max, nrbc_min, nrbc_max, d_dimer_min, d_dimer_max, fibrinogen_min, fibrinogen_max, thrombin_min, thrombin_max, inr_min, inr_max, pt_min, pt_max, ptt_min, ptt_max, alt_min, alt_max, alp_min, alp_max, ast_min, ast_max, amylase_min, amylase_max, bilirubin_total_min, bilirubin_total_max, bilirubin_direct_min, bilirubin_direct_max, bilirubin_indirect_min, bilirubin_indirect_max, ck_cpk_min, ck_cpk_max, ck_mb_min, ck_mb_max, ggt_min, ggt_max, ld_ldh_min, ld_ldh_max
- `concept.first_day_rrt`: subject_id, stay_id, dialysis_present, dialysis_active, dialysis_type
- `concept.first_day_sofa`: subject_id, hadm_id, stay_id, sofa, respiration, coagulation, liver, cardiovascular, cns, renal
- `concept.first_day_urine_output`: subject_id, stay_id, urineoutput
- `concept.first_day_vitalsign`: subject_id, stay_id, heart_rate_min, heart_rate_max, heart_rate_mean, sbp_min, sbp_max, sbp_mean, dbp_min, dbp_max, dbp_mean, mbp_min, mbp_max, mbp_mean, resp_rate_min, resp_rate_max, resp_rate_mean, temperature_min, temperature_max, temperature_mean, spo2_min, spo2_max, spo2_mean, glucose_min, glucose_max, glucose_mean
- `concept.first_day_weight`: subject_id, stay_id, weight_admit, weight, weight_min, weight_max

## measurement
- `concept.bg`: subject_id, hadm_id, charttime, specimen, so2, po2, pco2, fio2_chartevents, fio2, aado2, aado2_calc, pao2fio2ratio, ph, baseexcess, bicarbonate, totalco2, hematocrit, hemoglobin, carboxyhemoglobin, methemoglobin, chloride, calcium, temperature, potassium, sodium, lactate, glucose
- `concept.blood_differential`: subject_id, hadm_id, charttime, specimen_id, wbc, basophils_abs, eosinophils_abs, lymphocytes_abs, monocytes_abs, neutrophils_abs, basophils, eosinophils, lymphocytes, monocytes, neutrophils, atypical_lymphocytes, bands, immature_granulocytes, metamyelocytes, nrbc
- `concept.cardiac_marker`: subject_id, hadm_id, charttime, specimen_id, troponin_t, ck_mb, ntprobnp
- `concept.chemistry`: subject_id, hadm_id, charttime, specimen_id, albumin, globulin, total_protein, aniongap, bicarbonate, bun, calcium, chloride, creatinine, glucose, sodium, potassium
- `concept.coagulation`: subject_id, hadm_id, charttime, specimen_id, d_dimer, fibrinogen, thrombin, inr, pt, ptt
- `concept.complete_blood_count`: subject_id, hadm_id, charttime, specimen_id, hematocrit, hemoglobin, mch, mchc, mcv, platelet, rbc, rdw, rdwsd, wbc
- `concept.creatinine_baseline`: hadm_id, gender, age, scr_min, ckd, mdrd_est, scr_baseline
- `concept.enzyme`: subject_id, hadm_id, charttime, specimen_id, alt, alp, ast, amylase, bilirubin_total, bilirubin_direct, bilirubin_indirect, ck_cpk, ck_mb, ggt, ld_ldh
- `concept.gcs`: subject_id, stay_id, charttime, gcs, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable
- `concept.height`: subject_id, stay_id, charttime, height
- `concept.icp`: subject_id, stay_id, charttime, icp
- `concept.inflammation`: subject_id, hadm_id, charttime, specimen_id, crp
- `concept.oxygen_delivery`: subject_id, stay_id, charttime, o2_flow, o2_flow_additional, o2_delivery_device_1, o2_delivery_device_2, o2_delivery_device_3, o2_delivery_device_4
- `concept.rhythm`: subject_id, charttime, heart_rhythm, ectopy_type, ectopy_frequency, ectopy_type_secondary, ectopy_frequency_secondary
- `concept.urine_output`: stay_id, charttime, urineoutput
- `concept.urine_output_rate`: stay_id, charttime, weight, uo, urineoutput_6hr, urineoutput_12hr, urineoutput_24hr, uo_mlkghr_6hr, uo_mlkghr_12hr, uo_mlkghr_24hr, uo_tm_6hr, uo_tm_12hr, uo_tm_24hr
- `concept.ventilator_setting`: subject_id, stay_id, charttime, respiratory_rate_set, respiratory_rate_total, respiratory_rate_spontaneous, minute_volume, tidal_volume_set, tidal_volume_observed, tidal_volume_spontaneous, plateau_pressure, peep, fio2, flow_rate, ventilator_mode, ventilator_mode_hamilton, ventilator_type
- `concept.vitalsign`: subject_id, stay_id, charttime, heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, resp_rate, temperature, temperature_site, spo2, glucose

## medication
- `concept.acei`: subject_id, hadm_id, acei, starttime, stoptime
- `concept.antibiotic`: subject_id, hadm_id, stay_id, antibiotic, route, starttime, stoptime
- `concept.arb`: subject_id, hadm_id, arb, starttime, stoptime
- `concept.dobutamine`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.dopamine`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.epinephrine`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.milrinone`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.neuroblock`: stay_id, orderid, drug_rate, drug_amount, starttime, endtime
- `concept.norepinephrine`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.norepinephrine_equivalent_dose`: stay_id, starttime, endtime, norepinephrine_equivalent_dose
- `concept.nsaid`: subject_id, hadm_id, nsaid, starttime, stoptime
- `concept.phenylephrine`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime
- `concept.vasoactive_agent`: stay_id, starttime, endtime, dopamine, epinephrine, norepinephrine, phenylephrine, vasopressin, dobutamine, milrinone
- `concept.vasopressin`: stay_id, linkorderid, vaso_rate, vaso_amount, starttime, endtime

## organfailure
- `concept.kdigo_creatinine`: hadm_id, stay_id, charttime, creat, creat_low_past_48hr, creat_low_past_7day
- `concept.kdigo_stages`: subject_id, hadm_id, stay_id, charttime, creat_low_past_7day, creat_low_past_48hr, creat, aki_stage_creat, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr, aki_stage_uo, aki_stage_crrt, aki_stage, aki_stage_smoothed
- `concept.kdigo_uo`: stay_id, charttime, weight, urineoutput_6hr, urineoutput_12hr, urineoutput_24hr, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr, uo_tm_6hr, uo_tm_12hr, uo_tm_24hr
- `concept.meld`: subject_id, hadm_id, stay_id, meld_initial, meld, rrt, creatinine_max, bilirubin_total_max, inr_max, sodium_min

## score
- `concept.apsiii`: subject_id, hadm_id, stay_id, apsiii, apsiii_prob, hr_score, mbp_score, temp_score, resp_rate_score, pao2_aado2_score, hematocrit_score, wbc_score, creatinine_score, uo_score, bun_score, sodium_score, albumin_score, bilirubin_score, glucose_score, acidbase_score, gcs_score
- `concept.lods`: subject_id, hadm_id, stay_id, lods, neurologic, cardiovascular, renal, pulmonary, hematologic, hepatic
- `concept.oasis`: subject_id, hadm_id, stay_id, oasis, oasis_prob, age, age_score, preiculos, preiculos_score, gcs, gcs_score, heartrate, heart_rate_score, meanbp, mbp_score, resprate, resp_rate_score, temp, temp_score, urineoutput, urineoutput_score, mechvent, mechvent_score, electivesurgery, electivesurgery_score
- `concept.sapsii`: subject_id, hadm_id, stay_id, starttime, endtime, sapsii, sapsii_prob, age_score, hr_score, sysbp_score, temp_score, pao2fio2_score, uo_score, bun_score, wbc_score, potassium_score, sodium_score, bicarbonate_score, bilirubin_score, gcs_score, comorbidity_score, admissiontype_score
- `concept.sirs`: subject_id, hadm_id, stay_id, sirs, temp_score, heart_rate_score, resp_score, wbc_score
- `concept.sofa`: stay_id, hr, starttime, endtime, pao2fio2ratio_novent, pao2fio2ratio_vent, rate_epinephrine, rate_norepinephrine, rate_dopamine, rate_dobutamine, meanbp_min, gcs_min, uo_24hr, bilirubin_max, creatinine_max, platelet_min, respiration, coagulation, liver, cardiovascular, cns, renal, respiration_24hours, coagulation_24hours, liver_24hours, cardiovascular_24hours, cns_24hours, renal_24hours, sofa_24hours

## sepsis
- `concept.sepsis3`: subject_id, stay_id, antibiotic_time, culture_time, suspected_infection_time, sofa_time, sofa_score, respiration, coagulation, liver, cardiovascular, cns, renal, sepsis3
- `concept.suspicion_of_infection`: subject_id, stay_id, hadm_id, ab_id, antibiotic, antibiotic_time, suspected_infection, suspected_infection_time, culture_time, specimen, positive_culture

## treatment
- `concept.code_status`: subject_id, hadm_id, stay_id, charttime, fullcode, cmo, dni, dnr
- `concept.crrt`: stay_id, charttime, crrt_mode, access_pressure, blood_flow, citrate, current_goal, dialysate_fluid, dialysate_rate, effluent_pressure, filter_pressure, heparin_concentration, heparin_dose, hourly_patient_fluid_removal, prefilter_replacement_rate, postfilter_replacement_rate, replacement_fluid, replacement_rate, return_pressure, ultrafiltrate_output, system_active, clots, clots_increasing, clotted
- `concept.invasive_line`: stay_id, line_type, line_site, starttime, endtime
- `concept.rrt`: stay_id, charttime, dialysis_present, dialysis_active, dialysis_type
- `concept.ventilation`: stay_id, starttime, endtime, ventilation_status
