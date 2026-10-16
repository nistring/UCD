"""DesignSpec: the planner's output. Human-readable design fields + one analysis SQL.

All windows are in hours relative to the index time (time zero) of each unit.
`analysis_sql` must return ONE row per candidate unit with:
  unit_id, subject_key, index_time (timestamp), unit_end_time (timestamp, optional),
  one boolean column per population criterion (named as in population_criteria[].flag, applied in order),
  outcome (0/1, NULL = unknown), optional outcome_time_h / exposure_time_h (hours from index),
  and every column used by exposure / covariates / model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class PICO(BaseModel):
    population: str
    exposure_or_predictor: str
    comparator: str = ""
    outcome: str


class Criterion(BaseModel):
    flag: str                      # column name in analysis_sql output, e.g. incl_adult
    description: str


TYPE_ALIASES = {"continuous": "numeric", "numerical": "numeric", "number": "numeric", "float": "numeric",
                "integer": "numeric", "int": "numeric", "count": "numeric", "ordinal": "numeric", "score": "numeric",
                "boolean": "binary", "bool": "binary", "dichotomous": "binary", "indicator": "binary", "flag": "binary",
                "nominal": "categorical", "category": "categorical", "factor": "categorical", "string": "categorical"}


class Variable(BaseModel):
    column: str
    description: str
    type: Literal["numeric", "binary", "categorical"] = "numeric"
    window_start_h: Optional[float] = None
    window_end_h: Optional[float] = None
    reference_level: Optional[str] = None   # categorical reference

    @field_validator("type", mode="before")
    @classmethod
    def _type_alias(cls, v):
        """Common synonyms map to the three types (E-56: 'continuous' failed validation turn after turn)."""
        return TYPE_ALIASES.get(str(v).strip().lower(), v) if isinstance(v, str) else v


class Outcome(BaseModel):
    column: str = "outcome"
    description: str


class Model(BaseModel):
    type: Literal["logistic", "score_eval", "descriptive", "cox", "cohort", "weighted_binary"]
    # cohort: the cohort itself is the answer; weighted_binary: propensity-weighted risk difference (target trial)
    predictors: list[str] = Field(default_factory=list)       # logistic: columns (exposure first)
    primary_term: Optional[str] = None                        # logistic: column (or column=level) reported as main result
    score_columns: list[str] = Field(default_factory=list)    # score_eval: first = primary score
    value_column: Optional[str] = None                        # descriptive: summarized numeric column
    group_column: Optional[str] = None                        # descriptive: grouping column
    time_column: Optional[str] = None                         # cox: hours from index to event or censoring
    cluster_by_patient: bool = False                          # logistic/cox: robust SEs clustered on subject_key
                                                              # (keeps repeat units without dropping them)
    td_exposure_time_column: Optional[str] = None             # cox: hours from index to start of a time-dependent
                                                              #      binary exposure (NULL = never exposed); term "td_exposure"
    ps_covariates: list[str] = Field(default_factory=list)    # weighted_binary: propensity-model covariates
    weights: Literal["overlap", "iptw"] = "overlap"           # weighted_binary: overlap weights or IPTW
    horizon_days: Optional[float] = None                      # weighted_binary: outcome horizon (RMST up to it)
    bootstrap: int = 200                                      # weighted_binary: bootstrap replicates for the CIs


class Waiver(BaseModel):
    rule: str                      # e.g. V4; only tier-2 (contextual) rules can be waived
    protocol_items: list[str]      # ids from protocol_items.yaml that mandate the design element
    justification: str


class Sensitivity(BaseModel):
    name: str
    flag: str                      # boolean column; the main model is refitted on rows where it is true


class ProtocolMap(BaseModel):
    item: str                      # id from protocol_items.yaml, e.g. P3
    implemented_as: str            # flag/column/model element that implements it
    deviation: str = ""            # empty = implemented exactly; otherwise what differs and why


class DesignSpec(BaseModel):
    title: str
    question: str
    pico: PICO
    unit_of_analysis: Literal["icu_stay", "admission", "patient"]
    repeated_units: Literal["first_only", "all_independent", "one_per_patient_other"] = "first_only"
    index_time: str                                           # text definition of time zero
    prediction_time_h: float = 0.0                            # predictors are complete this many hours after index
    outcome_from_h: float = 0.0                               # outcomes are counted from this many hours after index
                                                              # (0 = from time zero; = prediction_time_h for a landmark)
    population_criteria: list[Criterion]
    exposure: Optional[Variable] = None
    outcome: Outcome
    covariates: list[Variable] = Field(default_factory=list)
    model: Model
    protocol_mapping: list[ProtocolMap] = Field(default_factory=list)
    waivers: list[Waiver] = Field(default_factory=list)
    sensitivity: list[Sensitivity] = Field(default_factory=list)
    shrink_justification: str = ""
    exposure_inclusive_at_t0: bool = True      # an exposure starting exactly at time zero counts (rule T3)
    changes_from_previous: str = ""
    analysis_sql: str

    @field_validator("population_criteria")
    @classmethod
    def _flags(cls, v):
        if not v:
            raise ValueError("at least one population criterion is required")
        return v

    def columns_needed(self) -> list[str]:
        outcome = [] if self.model.type == "cohort" else [self.outcome.column]   # a cohort task has no outcome
        cols = ["unit_id", "subject_key", "index_time"] + outcome + [c.flag for c in self.population_criteria]
        if self.exposure:
            cols.append(self.exposure.column)
        cols += [c.column for c in self.covariates]
        m = self.model
        cols += m.predictors + m.score_columns + m.ps_covariates + [
            x for x in (m.value_column, m.group_column, m.time_column, m.td_exposure_time_column) if x]
        cols += [s.flag for s in self.sensitivity]
        seen, out = set(), []
        for c in cols:
            c = c.split("=")[0]
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out


def load_spec(path: str | Path) -> DesignSpec:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:          # e.g. a stray cp1252/cp949 byte from a shell heredoc
        text = raw.decode("utf-8", errors="replace")
    data = yaml.safe_load(text)
    return DesignSpec.model_validate(data)


TEMPLATE = """\
title: ""
question: ""
pico: {population: "", exposure_or_predictor: "", comparator: "", outcome: ""}
unit_of_analysis: icu_stay          # icu_stay | admission | patient
repeated_units: first_only          # first_only | all_independent | one_per_patient_other
index_time: "ICU admission (icustays.intime)"
prediction_time_h: 0                # predictors complete this many hours after index (e.g. 24 for day-1 scores)
outcome_from_h: 0                   # outcomes counted from this many hours after index: 0 = from time zero,
                                    # = prediction_time_h for a landmark (then exclude earlier events with a flag)
population_criteria:                # applied in order -> flow diagram; each flag is a boolean column in analysis_sql
  - {flag: incl_adult, description: "age >= 18 at admission"}
exposure: {column: "", description: "", type: binary, window_start_h: 0, window_end_h: 24}
outcome: {column: outcome, description: ""}
covariates:
  - {column: age, description: "age at admission (years)", type: numeric, window_start_h: 0, window_end_h: 0}
model: {type: logistic, predictors: [], primary_term: ""}
# weighted_binary example: {type: weighted_binary, ps_covariates: [age, sofa_nonrenal], weights: overlap,
#                           horizon_days: 28, time_column: outcome_time_h}   (exposure: the binary treatment)
# model types: logistic | score_eval | descriptive | cox
# cox example: {type: cox, time_column: followup_h, td_exposure_time_column: rbc_time_h,
#               predictors: [age, sofa_d1], primary_term: td_exposure}
waivers: []                         # [{rule: V4, protocol_items: [P3, P9], justification: "..."}] - tier-2 rules only
sensitivity: []                     # [{name: "24 h landmark", flag: incl_alive_24h}] - main model refitted on flag=true
protocol_mapping:                   # one entry per item in protocol_items.yaml
  - {item: P1, implemented_as: "incl_adult", deviation: ""}
shrink_justification: ""
limitations: []
changes_from_previous: ""
analysis_sql: |
  SELECT ...
"""
