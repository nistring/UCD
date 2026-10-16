# External reference papers: verification log

Verified 2026-10-16. Bibliographic data (authors, journal, volume/pages, DOI, PMID, PMCID) for every paper came from
NCBI E-utilities (`efetch.fcgi?db=pubmed&id=...&rettype=abstract`). The pubmed.ncbi.nlm.nih.gov and pmc.ncbi.nlm.nih.gov web
pages were blocked by a cookie/captcha wall, so open-access full text was taken from Europe PMC
(`https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML`). Open-access status was checked with
Europe PMC search and Unpaywall. None of the 7 papers uses MIMIC data.

Rule applied: every number in `papers.yaml` has a verbatim quote. A value that is not in the accessible text is `null`.
A design item that could not be verified in the paper is marked "not verified".

## t1_aki: Hoste 2015, AKI-EPI (kept)
- Sources: E-utilities PMID 26162677. Springer page `link.springer.com/article/10.1007/s00134-015-3934-7` showed the abstract only ("preview of subscription content"). Unpaywall and Europe PMC: not open access.
- Verified: citation, DOI, PMID, n=1802, AKI 57.3% (55.0-59.6), adjusted ORs for stages 1-3, eGFR at discharge.
- Not available (left null or marked unverified): overall hospital mortality, stage distribution, baseline-creatinine rule, covariates, exclusion criteria, enrolment dates.
- Why kept: prospective, multinational, full KDIGO (creatinine + UO), first ICU week, adjusted ORs by stage. It matches the question exactly.

## t2_sofa: Raith 2017, JAMA, ANZICS (kept)
- Sources: E-utilities PMID 28114553; full text at `https://jamanetwork.com/journals/jama/fullarticle/2598267` (main text read, supplement not read).
- Verified: n=184,875; mortality 18.7%; crude AUROCs (SOFA 0.753, SIRS 0.589, qSOFA 0.607, 99% CI); adjusted AUROCs (SOFA 0.815, SIRS 0.755, qSOFA 0.762). Also verified: age >=17, exclusion of same-episode readmissions and transfers, baseline SOFA = 0, first-24 h values, missing components scored 0, no vasopressor dose for cardiovascular SOFA, covariates of the baseline-risk model.
- Note: the paper is internally inconsistent for adjusted qSOFA (Table 2 gives 0.762, Figure 3B gives 0.763). The yaml records 0.762.

## t3_dysnatremia: Funk 2010, Intensive Care Med (kept, alternative considered)
- Sources: E-utilities PMID 19847398 (full abstract). Springer full text paywalled; Unpaywall: not OA. A web search showed a secondary commentary (clinician.com, HTTP 403 on fetch) that gives exclusion counts (8509 readmissions, 3292 <18 y, 13,416 missing outcome/SAPS II/sodium). These are recorded as "not verified" and not used as numbers. The commentary quotes severe hypernatremia OR as 3.62; the abstract says 3.64, and the yaml uses the abstract.
- Verified: n=151,486; 77 Austrian ICUs, 1998-2007; prevalence of each sodium band; all 6 adjusted ORs with 95% CIs.
- Not available: overall hospital mortality (null), exact covariates, sodium timing rule.
- Replacement considered: Darmon M et al., "Prognostic consequences of borderline dysnatremia", Crit Care 2013;17:R12 (doi 10.1186/cc11937; OUTCOMEREA, France, open access). Rejected: smaller (~11k), and Funk is larger, in a higher-impact journal, inside the 2010-2021 window, and fits the question exactly (sodium at ICU admission, graded bands, hospital mortality, adjusted ORs). The only weakness is the paywalled full text.

## t4_ne_timing: Bai 2014, Crit Care (kept, with alternate reference added)
- Sources: E-utilities PMID 25277635; full text from Europe PMC PMC4194405.
- Verified: n=213; 28-day mortality 37.6%; mean time to NE 3.1 ± 2.5 h; unadjusted OR 1.20/h (1.07-1.35); adjusted OR 1.392/h (1.138-1.702, Table 5); 29.1% vs 43.3% (<2 h vs >=2 h); 5.3%/h; group sizes 86/127; septic shock definition; setting (two surgical ICUs at Jinling Hospital, Nanjing, 2011-2012); model covariates.
- Concern: small, single-centre, surgical ICU, very large effect. Bai is kept as primary because it is the specific norepinephrine-timing study named in the question. I added `alternate_reference`: Beck V et al., Crit Care 2014;18:R97 (PMID 24887489, PMC4075345; CATSS, 28 ICUs in Canada/USA/Saudi Arabia, n=6514). Beck reports a weak adjusted effect, OR 1.02/h (1.01-1.03), hospital mortality 53%, and median time to vasopressor 3 h (IQR 1-7.1). Its exposure is any vasopressor/inotrope, not norepinephrine only. Use both to bracket a MIMIC replication.

## t5_readmission: Kramer, Higgins, Zimmerman 2012, Crit Care Med (kept)
- Sources: E-utilities PMID 21926603 (full abstract); BHS scholarly commons record (`scholarlycommons.libraryinfo.bhs.org/all_works/1406`, abstract only); LWW full text returned HTTP 402; Unpaywall: not OA.
- Verified: authors (Kramer AA, Higgins TL, Zimmerman JE), n=229,375 first admissions, 97 units / 35 US hospitals, 2001-2009, readmission 6.1% (13,980), adjusted four-fold hospital mortality and 2.5-fold LOS, and the list of predictor domains.
- Not available: individual predictor ORs, model C-statistic, exact exclusions, database name. These are recorded as null or "not verified".
- Why kept: it is the standard large multicentre US reference for same-hospitalization ICU readmission, and it is not MIMIC.

## t6_transfusion: Vincent 2018, ICON, Crit Care (kept)
- Sources: E-utilities PMID 29673409; full text from Europe PMC PMC5909204.
- Verified: n=9553 (of 10,069); 730 ICUs in 84 countries, 8-18 May 2012; hospital mortality 22.3%; transfused 26.3%; trigger 8.3 ± 1.7 g/dL (and 8.3 ± 2.2 on the admission day); crude mortality 30.0% vs 19.6%; adjusted HR 0.98 (0.96-1.00), from an extended Cox model with time-dependent transfusion; 62% of patients with Hb <7 transfused; median 3 units; full covariate list.

## t7_meld: Theocharidou 2014, Am J Gastroenterol (kept)
- Sources: E-utilities PMID 24492755; full text from Europe PMC PMC3978197.
- Verified: n=635, single centre, Royal Free, 1989-2012; exclusions (acute liver failure, post-transplant, postoperative hepatobiliary); in-hospital mortality 52.3%, ICU mortality 30.2%; scores taken on the ICU admission day. Table 5 AUROCs (training/validation): MELD 0.787/0.749, MELD-Na 0.762/0.716, SOFA 0.810/0.785. 2005-2012 subset (n=306): MELD 0.73, MELD-Na 0.71, SOFA 0.74, CLIF-SOFA 0.75. Median MELD 26 vs 18; MELD cut-off 21.
- No AUROC CIs are reported (ci: null). MELD-Na is evaluated, so the paper covers both scores in the question.
- Limitation noted in yaml: single specialist liver ICU and an old era (from 1989). It is kept because it reports MELD and MELD-Na side by side in ICU cirrhosis patients with in-hospital mortality, and its full text is open access.

## Process note
- The Unpaywall API requires an email parameter; the user's email was used in those 4 lookup calls (DOIs only, no other data).
