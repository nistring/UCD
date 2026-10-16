You are an external evaluator, not part of the agent. Project root: the repository root. Topic: {topic}.

**Task.** Score how closely each design iteration the agent produced matches the reference study's design.
- Read the reference entry `{topic}` in `eval/papers.yaml`, especially `design_checklist` and `population`, `exposure_definition`, `outcome_definition` and `model`.
- For every run directory `runs/ucd/{topic}/r*/`, read each `mimic/iter_*/spec.yaml`.
- Skip a spec that is empty or invalid and score it 1.0.
- Do NOT read data files and do NOT run anything except Python to merge JSON.

**Scoring.** Judge each checklist item as one of:
- 1 = matched (implemented as the reference did it);
- 0.5 = partly matched or approximated with a stated reason;
- 0 = not matched or absent.

Then E_qual = 1 - mean(item scores), so 0 is a perfect design match. Judge only the design (cohort, definitions, windows, model), not the numbers. Be consistent across runs: use the same standard for every iteration.

**Output.** Write `eval/qual/{topic}.json` as an object of this form:
```
{"<topic>/<run>/mimic/<iter>": {"E_qual": <float>, "items": [<0|0.5|1>, ...], "notes": "<one line: main mismatches>"}, ...}
```
Use keys like "t2_sofa/r1/mimic/3". Your final reply is under 100 words: mean E_qual per run, and the most common mismatches.
