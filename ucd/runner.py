"""Deterministic runner: DesignSpec -> result card (aggregates only, counts 1-9 suppressed).

Models (clinician-explainable only): logistic regression (odds ratios), evaluation of an existing score
(AUROC + calibration of a one-variable logistic recalibration), descriptive (medians by group).
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from ucd.gate import K_MIN, Gate, GateError
from ucd.spec import DesignSpec

warnings.filterwarnings("ignore")


def sup(n: Any) -> Any:
    """Suppress counts 1..K_MIN-1."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return n
    return f"<{K_MIN}" if 0 < n < K_MIN else n


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _r(x: Any, nd: int = 4) -> Any:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return None if math.isnan(xf) or math.isinf(xf) else round(xf, nd)


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda v: False if pd.isna(v) else bool(v)).astype(bool)


# ---------------------------------------------------------------------------------------------- variables
def describe_var(s: pd.Series, y: pd.Series, vtype: str) -> dict[str, Any]:
    out: dict[str, Any] = {"missing_frac": _r(s.isna().mean(), 3)}
    if vtype == "numeric":
        x = pd.to_numeric(s, errors="coerce")
        out["missing_frac"] = _r(x.isna().mean(), 3)
        xv = x.dropna()
        if len(xv) >= K_MIN:
            q = xv.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = q[0.75] - q[0.25]
            out.update(p01=_r(q[0.01]), q1=_r(q[0.25]), median=_r(q[0.5]), q3=_r(q[0.75]), p99=_r(q[0.99]),
                       mean=_r(xv.mean()), sd=_r(xv.std()))
            out["extreme_frac"] = _r(((xv - q[0.5]).abs() > 10 * max(iqr, 1e-9)).mean(), 4) if iqr > 0 else 0.0
            out["zero_variance"] = bool(xv.nunique() <= 1)
            for lab, m in (("median_outcome1", y == 1), ("median_outcome0", y == 0)):
                v = x[m].dropna()
                out[lab] = _r(v.median()) if len(v) >= K_MIN else None
    else:
        vc = s.astype("string").fillna("<missing>").value_counts()
        levels = {}
        small = int(vc[vc < K_MIN].sum())
        for lev, n in vc[vc >= K_MIN].head(20).items():
            m = s.astype("string").fillna("<missing>") == lev
            ev = int(y[m].sum())
            levels[str(lev)] = {"n": int(n), "outcome_rate": _r(ev / n, 3)}
        if small:
            levels["<levels with n<10, pooled>"] = {"n": sup(small), "outcome_rate": None}
        out["levels"] = levels
        out["n_levels"] = int(vc.size)
        out["zero_variance"] = bool(s.dropna().nunique() <= 1)
    return out


def _design_matrix(df: pd.DataFrame, spec: DesignSpec, predictors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    types = {v.column: v for v in spec.covariates}
    if spec.exposure:
        types[spec.exposure.column] = spec.exposure
    parts = []
    for c in predictors:
        v = types.get(c)
        vtype = v.type if v else ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical")
        if vtype == "categorical":
            s = df[c].astype("string")
            ref = v.reference_level if v and v.reference_level else (s.value_counts().index[0] if s.notna().any() else None)
            d = pd.get_dummies(s, prefix=c, prefix_sep="=", dtype=float)
            if ref is not None and f"{c}={ref}" in d.columns:
                d = d.drop(columns=[f"{c}={ref}"])
            d[s.isna()] = np.nan
            parts.append(d)
        else:
            parts.append(pd.to_numeric(df[c].astype("float64") if df[c].dtype == bool else df[c],
                                       errors="coerce").rename(c).to_frame())
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return X, list(X.columns)


def _auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _calib(y: np.ndarray, p: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        slope = float(fit.params[1])
        itl = sm.GLM(y, np.ones_like(lp), offset=lp, family=sm.families.Binomial()).fit()
        return float(itl.params[0]), slope
    except Exception:
        return None, None


def fit_logistic(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    preds = [p.split("=")[0] for p in spec.model.predictors]
    X, names = _design_matrix(df, spec, preds)
    data = pd.concat([X, y.rename("__y")], axis=1).dropna()
    n_used = len(data)
    out: dict[str, Any] = {"n_used": sup(n_used), "n_dropped_incomplete": sup(len(df) - n_used),
                           "complete_case_drop_frac": _r(1 - n_used / max(len(df), 1), 3),
                           "n_params": len(names)}
    if n_used < K_MIN or data["__y"].nunique() < 2:
        out["error"] = "too few complete rows or outcome has one level"
        return out
    ev = int(min(data["__y"].sum(), n_used - data["__y"].sum()))
    out["epv"] = _r(ev / max(len(names), 1), 1)
    Xc = sm.add_constant(data[names].astype(float), has_constant="add")
    try:
        if spec.model.cluster_by_patient:
            groups = pd.factorize(df.loc[data.index, "subject_key"].astype(str))[0]
            fit = sm.Logit(data["__y"].astype(float), Xc).fit(disp=0, maxiter=200, cov_type="cluster",
                                                                cov_kwds={"groups": groups})
            out["se"] = "cluster-robust by patient"
        else:
            fit = sm.Logit(data["__y"].astype(float), Xc).fit(disp=0, maxiter=200)
        conv = bool(fit.mle_retvals.get("converged", True))
    except Exception as e:  # perfect separation, singular matrix
        out["error"] = f"model failed: {type(e).__name__}: {str(e)[:200]}"
        out["converged"] = False
        return out
    ci = fit.conf_int()
    terms = []
    for nm in names:
        b = fit.params[nm]
        terms.append({"term": nm, "or": _r(math.exp(b), 3) if abs(b) < 50 else None,
                      "ci_low": _r(math.exp(ci.loc[nm, 0]), 3) if abs(ci.loc[nm, 0]) < 50 else None,
                      "ci_high": _r(math.exp(ci.loc[nm, 1]), 3) if abs(ci.loc[nm, 1]) < 50 else None,
                      "p": _r(fit.pvalues[nm], 4)})
    out["converged"] = conv
    out["separation_suspect"] = bool((fit.params.abs() > 15).any() or (fit.bse > 50).any())
    out["terms"] = terms
    out["intercept"] = _r(fit.params["const"], 3)
    yv = data["__y"].astype(int).to_numpy()
    out["auroc_apparent"] = _r(_auc(yv, fit.predict(Xc)), 3)
    # 5-fold CV: out-of-sample discrimination + calibration of the same specification
    try:
        oof = np.zeros(len(yv))
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        Xa = data[names].astype(float).to_numpy()
        for tr, te in skf.split(Xa, yv):
            m = LogisticRegression(C=np.inf, max_iter=2000).fit(Xa[tr], yv[tr])
            oof[te] = m.predict_proba(Xa[te])[:, 1]
        out["auroc_cv"] = _r(_auc(yv, oof), 3)
        itl, slope = _calib(yv, oof)
        out["cv_calibration_intercept"], out["cv_calibration_slope"] = _r(itl, 3), _r(slope, 3)
        out["brier_cv"] = _r(np.mean((oof - yv) ** 2), 4)
    except Exception as e:
        out["cv_error"] = str(e)[:200]
    # primary term
    pt = spec.model.primary_term
    if pt:
        cand = [t for t in terms if t["term"] == pt] or [t for t in terms if t["term"].split("=")[0] == pt]
        if cand:
            out["primary"] = cand[0]
            if len(cand) > 1:
                out["primary_all_levels"] = cand
    return out


def eval_scores(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {"scores": []}
    rng = np.random.default_rng(0)
    first = pd.to_numeric(df[spec.model.score_columns[0]], errors="coerce") if spec.model.score_columns else None
    for k, sc in enumerate(spec.model.score_columns):
        x = pd.to_numeric(df[sc], errors="coerce")
        m = x.notna() & y.notna()
        xv, yv = x[m].to_numpy(float), y[m].astype(int).to_numpy()
        r: dict[str, Any] = {"score": sc, "n_used": sup(int(m.sum())), "missing_frac": _r(1 - m.mean(), 3)}
        both = x.notna() & first.notna()
        if k and both.sum() >= K_MIN:                 # share of units where this score equals the first one (M6)
            r["same_as_first"] = _r(float((x[both] == first[both]).mean()), 3)
        auc = _auc(yv, xv) if len(yv) >= K_MIN else None
        r["auroc"] = _r(auc, 3)
        if auc is not None:
            boots = []
            for _ in range(300):
                idx = rng.integers(0, len(yv), len(yv))
                a = _auc(yv[idx], xv[idx])
                if a is not None:
                    boots.append(a)
            r["auroc_ci"] = [_r(np.percentile(boots, 2.5), 3), _r(np.percentile(boots, 97.5), 3)]
            # one-variable logistic recalibration: OR per score point, outcome rate by score band
            import statsmodels.api as sm
            try:
                fit = sm.Logit(yv, sm.add_constant(xv)).fit(disp=0)
                r["or_per_point"] = _r(math.exp(fit.params[1]), 3)
            except Exception:
                r["or_per_point"] = None
            bands = pd.qcut(pd.Series(xv), q=min(5, max(1, len(np.unique(xv)))), duplicates="drop")
            tab = pd.DataFrame({"band": bands.astype(str), "y": yv}).groupby("band", observed=True)["y"].agg(["count", "mean"])
            r["outcome_rate_by_band"] = {k: {"n": sup(v["count"]), "rate": _r(v["mean"], 3) if v["count"] >= K_MIN else None}
                                         for k, v in tab.iterrows()}
        out["scores"].append(r)
    if out["scores"]:
        out["primary"] = {"term": out["scores"][0]["score"], "auroc": out["scores"][0].get("auroc"),
                          "ci": out["scores"][0].get("auroc_ci")}
    return out


def describe_groups(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    vc, gc = spec.model.value_column, spec.model.group_column
    out: dict[str, Any] = {}
    if vc:
        x = pd.to_numeric(df[vc], errors="coerce")
        groups = df[gc].astype("string").fillna("<missing>") if gc else pd.Series("all", index=df.index)
        res = {}
        for g, idx in groups.groupby(groups).groups.items():
            v = x.loc[idx].dropna()
            res[str(g)] = {"n": sup(len(idx)), "n_with_value": sup(len(v)),
                           "median": _r(v.median()) if len(v) >= K_MIN else None,
                           "q1": _r(v.quantile(.25)) if len(v) >= K_MIN else None,
                           "q3": _r(v.quantile(.75)) if len(v) >= K_MIN else None,
                           "outcome_rate": _r(y.loc[idx].mean(), 3) if len(idx) >= K_MIN else None}
        out["by_group"] = res
        allv = x.dropna()
        out["primary"] = {"term": f"median {vc}", "value": _r(allv.median()) if len(allv) >= K_MIN else None}
    if spec.model.predictors:
        out["logistic"] = fit_logistic(df, spec, y)
    return out


def fit_cox(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    """Cox proportional hazards (statsmodels PHReg). Optional time-dependent binary exposure via counting-process
    rows: [outcome_from, t_exp) unexposed and [t_exp, t_end] exposed. Time scale: hours from index."""
    import statsmodels.api as sm
    from statsmodels.duration.hazard_regression import PHReg

    m = spec.model
    t_end = pd.to_numeric(df[m.time_column], errors="coerce")
    start0 = float(spec.outcome_from_h or 0.0)
    preds = [p.split("=")[0] for p in m.predictors]
    X, names = _design_matrix(df, spec, preds) if preds else (pd.DataFrame(index=df.index), [])
    base = pd.concat([X, y.rename("__event"), t_end.rename("__stop"), df["subject_key"].rename("__subject")], axis=1)
    if m.td_exposure_time_column:
        base["__texp"] = pd.to_numeric(df[m.td_exposure_time_column], errors="coerce")
    base = base.dropna(subset=names + ["__event", "__stop"])
    base = base[base["__stop"] > start0]
    out: dict[str, Any] = {"n_used": sup(len(base)), "n_dropped_incomplete": sup(len(df) - len(base)),
                           "complete_case_drop_frac": _r(1 - len(base) / max(len(df), 1), 3)}
    rows = []
    for idx, r in base.iterrows():
        cov = {n: r[n] for n in names}
        cov["__subject"] = r["__subject"]
        te = r.get("__texp") if m.td_exposure_time_column else None
        stop, ev = float(r["__stop"]), int(r["__event"])
        if te is None or pd.isna(te) or te >= stop:
            rows.append({**cov, "td_exposure": 0.0, "__start": start0, "__stop": stop, "__event": ev})
        elif te <= start0:
            rows.append({**cov, "td_exposure": 1.0, "__start": start0, "__stop": stop, "__event": ev})
        else:
            rows.append({**cov, "td_exposure": 0.0, "__start": start0, "__stop": float(te), "__event": 0})
            rows.append({**cov, "td_exposure": 1.0, "__start": float(te), "__stop": stop, "__event": ev})
    L = pd.DataFrame(rows)
    terms_names = names + (["td_exposure"] if m.td_exposure_time_column else [])
    if L.empty or L["__event"].sum() < K_MIN or not terms_names:
        out["error"] = "too few events or no predictors for the Cox model"
        return out
    out["n_intervals"] = int(len(L))
    if m.td_exposure_time_column:
        out["n_ever_exposed"] = sup(int((base["__texp"].notna() & (base["__texp"] < base["__stop"])).sum()))
        out["person_hours_exposed"] = _r(((L["__stop"] - L["__start"]) * L["td_exposure"]).sum(), 0)
        out["person_hours_unexposed"] = _r(((L["__stop"] - L["__start"]) * (1 - L["td_exposure"])).sum(), 0)
    ev_n = int(L["__event"].sum())
    out["events"] = sup(ev_n)
    out["epv"] = _r(ev_n / max(len(terms_names), 1), 1)
    Xc = L[terms_names].astype(float)
    try:
        mod = PHReg(L["__stop"].to_numpy(), Xc.to_numpy(), status=L["__event"].to_numpy(),
                    entry=L["__start"].to_numpy(), ties="efron")
        if m.cluster_by_patient:
            fit = mod.fit(disp=0, groups=pd.factorize(L["__subject"].astype(str))[0])
            out["se"] = "cluster-robust by patient"
            if not np.all(np.isfinite(np.asarray(fit.bse))):     # robust SE undefined with delayed-entry rows
                fit = mod.fit(disp=0)
                out["se"] = "model-based (cluster-robust SE unavailable with time-dependent rows)"
        else:
            fit = mod.fit(disp=0)
    except Exception as e:
        out["error"] = f"Cox model failed: {type(e).__name__}: {str(e)[:200]}"
        out["converged"] = False
        return out
    params, bse = np.asarray(fit.params), np.asarray(fit.bse)
    terms = []
    for i, nm in enumerate(terms_names):
        b, se = params[i], bse[i]
        terms.append({"term": nm, "hr": _r(math.exp(b), 3) if abs(b) < 50 else None,
                      "ci_low": _r(math.exp(b - 1.96 * se), 3) if abs(b) < 50 else None,
                      "ci_high": _r(math.exp(b + 1.96 * se), 3) if abs(b) < 50 else None,
                      "p": _r(2 * (1 - __import__("scipy").stats.norm.cdf(abs(b / se))), 4) if se > 0 else None})
    out["terms"] = terms
    out["converged"] = True
    out["separation_suspect"] = bool((np.abs(params) > 15).any() or (bse > 50).any())
    pt = spec.model.primary_term
    if pt:
        cand = [t for t in terms if t["term"] == pt] or [t for t in terms if t["term"].split("=")[0] == pt]
        if cand:
            out["primary"] = cand[0]
    return out


def _wkm_rmst(t: np.ndarray, e: np.ndarray, w: np.ndarray, tau: float) -> float:
    """Restricted mean survival time up to tau from a weighted Kaplan-Meier curve."""
    order = np.argsort(t, kind="mergesort")
    t, e, w = t[order], e[order], w[order]
    at_risk, s, last, area = w.sum(), 1.0, 0.0, 0.0
    for time in np.unique(t[t <= tau]):
        here = t == time
        area += s * (time - last)
        d = w[here & (e == 1)].sum()
        if at_risk > 0:
            s *= 1 - d / at_risk
        at_risk -= w[here].sum()
        last = time
    return float(area + s * (tau - last))


def fit_weighted_binary(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    """Target-trial-emulation estimate (0.10.0): logistic propensity score for the binary exposure on
    model.ps_covariates, overlap weights (or IPTW), the weighted risk difference (primary), the weighted
    restricted-mean-survival-time difference over model.horizon_days when model.time_column is given, percentile
    bootstrap CIs with the propensity model refitted in each resample, and covariate balance (max |SMD|).
    Missing covariates get median imputation plus a missing indicator, a simplification of multiple imputation
    that the card states. Aggregates only."""
    import statsmodels.api as sm
    m = spec.model
    if not spec.exposure:
        return {"error": "weighted_binary needs a binary exposure (spec.exposure)"}
    a = pd.to_numeric(df[spec.exposure.column].astype("float64") if df[spec.exposure.column].dtype == bool
                      else df[spec.exposure.column], errors="coerce")
    X, _ = _design_matrix(df, spec, list(m.ps_covariates))
    for c in list(X.columns):
        if X[c].isna().any():
            X[f"{c}__missing"] = X[c].isna().astype(float)
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    ok = (a.isin([0, 1]) & y.isin([0, 1])).values
    A, Y, Xv = a.values[ok].astype(int), y.values[ok].astype(int), X.values[ok].astype(float)
    n, n1 = len(A), int(A.sum())
    if n1 < K_MIN or n - n1 < K_MIN:
        return {"error": "fewer than 10 units in an exposure group"}
    tau = (float(m.horizon_days) * 24.0) if m.horizon_days else None
    T = None
    if m.time_column and tau:                          # follow-up from the landmark, in hours, capped at the horizon
        tt = pd.to_numeric(df[m.time_column], errors="coerce").values[ok] - float(spec.outcome_from_h or 0)
        T = np.clip(np.nan_to_num(tt, nan=tau), 0, tau)

    def estimate(idx: np.ndarray) -> tuple[float, Optional[float], np.ndarray]:
        Xc = sm.add_constant(Xv[idx], has_constant="add")
        ps = np.clip(sm.Logit(A[idx], Xc).fit(disp=0, maxiter=200).predict(Xc), 0.01, 0.99)
        aa, yy = A[idx], Y[idx]
        w = aa * (1 - ps) + (1 - aa) * ps if m.weights == "overlap" else aa / ps + (1 - aa) / (1 - ps)
        rd = (w * yy * aa).sum() / (w * aa).sum() - (w * yy * (1 - aa)).sum() / (w * (1 - aa)).sum()
        rm = None
        if T is not None:
            tt = T[idx]
            rm = (_wkm_rmst(tt[aa == 1], yy[aa == 1], w[aa == 1], tau)
                  - _wkm_rmst(tt[aa == 0], yy[aa == 0], w[aa == 0], tau)) / 24.0      # days
        return float(rd), rm, w

    try:
        rd, rm, w = estimate(np.arange(n))
    except Exception as e:
        return {"error": f"propensity model failed: {type(e).__name__}: {str(e)[:200]}"}
    rng = np.random.default_rng(20261017)
    boot_rd, boot_rm = [], []
    for _ in range(max(0, int(m.bootstrap))):
        idx = rng.integers(0, n, n)
        if A[idx].sum() < 2 or (1 - A[idx]).sum() < 2:
            continue
        try:
            b_rd, b_rm, _ = estimate(idx)
        except Exception:
            continue
        boot_rd.append(b_rd)
        if b_rm is not None:
            boot_rm.append(b_rm)

    def ci(v: list[float]) -> list[Any]:
        return [_r(float(np.percentile(v, 2.5))), _r(float(np.percentile(v, 97.5)))] if len(v) >= 20 else [None, None]

    def max_smd(weights: np.ndarray) -> Optional[float]:
        out = []
        for j in range(Xv.shape[1]):
            x = Xv[:, j]
            m1 = np.average(x[A == 1], weights=weights[A == 1]); m0 = np.average(x[A == 0], weights=weights[A == 0])
            v1 = np.average((x[A == 1] - m1) ** 2, weights=weights[A == 1])
            v0 = np.average((x[A == 0] - m0) ** 2, weights=weights[A == 0])
            sd = math.sqrt((v1 + v0) / 2)
            if sd > 0:
                out.append(abs(m1 - m0) / sd)
        return _r(max(out), 3) if out else None

    def ess(weights: np.ndarray) -> Any:
        return sup(int(round(weights.sum() ** 2 / (weights ** 2).sum())))

    lo, hi = ci(boot_rd)
    res = {"type": "weighted_binary", "weights": m.weights, "n": sup(n), "n_exposed": sup(n1),
           "crude": {"risk_exposed": _r(Y[A == 1].mean()), "risk_unexposed": _r(Y[A == 0].mean())},
           "weighted": {"risk_exposed": _r((w * Y * A).sum() / (w * A).sum()),
                        "risk_unexposed": _r((w * Y * (1 - A)).sum() / (w * (1 - A)).sum())},
           "primary": {"term": "weighted risk difference (exposed - unexposed)", "value": _r(rd),
                       "ci_low": lo, "ci_high": hi},
           "balance": {"max_abs_smd_before": max_smd(np.ones(n)), "max_abs_smd_after": max_smd(w)},
           "ess": {"exposed": ess(w[A == 1]), "unexposed": ess(w[A == 0])},
           "bootstrap": {"replicates": len(boot_rd)},
           "missing_covariates": "median imputation with missing indicators (not multiple imputation)",
           "weighted_cox": "not available in this environment (supportive estimate only)"}
    if rm is not None:
        rlo, rhi = ci(boot_rm)
        res["rmst_difference_days"] = {"value": _r(rm, 3), "ci_low": rlo, "ci_high": rhi, "horizon_days": m.horizon_days}
    return res


def fit_main(df: pd.DataFrame, spec: DesignSpec, y: pd.Series) -> dict[str, Any]:
    t = spec.model.type
    if t == "weighted_binary":
        return fit_weighted_binary(df, spec, y)
    if t == "logistic":
        return fit_logistic(df, spec, y)
    if t == "score_eval":
        return eval_scores(df, spec, y)
    if t == "cox":
        return fit_cox(df, spec, y)
    return describe_groups(df, spec, y)


# ---------------------------------------------------------------------------------------------- main
def run(spec: DesignSpec, gate: Gate) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:
    """Returns (card, cohort). The cohort DataFrame never leaves the runner process except as unit_ids
    written by the CLI to the hidden evaluation folder."""
    card: dict[str, Any] = {"dataset": gate.dataset, "title": spec.title}
    try:
        raw = gate.materialize(spec.analysis_sql)
    except GateError as e:
        card["error"] = str(e)
        return card, None
    raw.columns = [str(c) for c in raw.columns]
    missing = [c for c in spec.columns_needed() if c not in raw.columns]
    if missing:
        card["error"] = f"analysis_sql output lacks required columns: {missing}; it returned {list(raw.columns)[:40]}"
        return card, None

    # flow
    df = raw
    flow = [{"step": "rows returned by analysis_sql", "n_units": sup(len(df)), "n_patients": sup(df["subject_key"].nunique())}]
    for c in spec.population_criteria:
        df = df[_to_bool(df[c.flag])]
        flow.append({"step": f"{c.flag}: {c.description}", "n_units": sup(len(df)), "n_patients": sup(df["subject_key"].nunique())})
    card["flow"] = flow
    n0, n = len(raw), len(df)
    card["n_start"], card["n_final"] = sup(n0), sup(n)
    card["retained_frac"] = _r(n / max(n0, 1), 3)
    card["empty"] = n == 0                                  # yes/no only: an empty cohort can be the answer
    if n < K_MIN:
        if spec.model.type == "cohort":                     # a cohort task: an empty or tiny cohort is a result
            card["model"], card["outcome"] = {"type": "cohort"}, {"events": None, "n": sup(n)}
            return card, df
        card["error"] = "final cohort has fewer than 10 units"
        return card, df

    # unit structure
    card["duplicate_unit_ids"] = sup(int(df["unit_id"].duplicated().sum()))
    n_pat = df["subject_key"].nunique()
    card["n_patients"] = sup(n_pat)
    card["units_per_patient_gt1_frac"] = _r(1 - n_pat / n, 3)
    if "unit_end_time" in df.columns:
        d = df[["subject_key", "index_time", "unit_end_time"]].copy()
        d["index_time"] = pd.to_datetime(d["index_time"], errors="coerce")
        d["unit_end_time"] = pd.to_datetime(d["unit_end_time"], errors="coerce")
        d = d.sort_values(["subject_key", "index_time"])
        prev_end = d.groupby("subject_key")["unit_end_time"].shift()
        gap_h = (d["index_time"] - prev_end).dt.total_seconds() / 3600
        card["contiguous_with_previous_unit_frac"] = _r((gap_h.abs() < 1).mean(), 4)
    if spec.model.type == "cohort":                         # no outcome, no model: flow, units and Table 1
        nil = pd.Series(-1, index=df.index)
        card["table1"] = {v.column: describe_var(df[v.column], nil, v.type)
                          for v in ([spec.exposure] if spec.exposure else []) + spec.covariates if v.column in df.columns}
        card["model"], card["outcome"] = {"type": "cohort"}, {"events": None, "n": sup(n)}
        return card, df
    # outcome
    y = pd.to_numeric(df[spec.outcome.column], errors="coerce")
    card["outcome_missing_frac"] = _r(y.isna().mean(), 3)
    k = int((y == 1).sum())
    nn = int(y.notna().sum())
    lo, hi = wilson(k, nn)
    card["outcome"] = {"events": sup(k), "n": sup(nn), "rate": _r(k / max(nn, 1), 4), "ci": [_r(lo), _r(hi)]}
    # timing checks
    timing: dict[str, Any] = {}
    if "outcome_time_h" in df.columns:
        ot = pd.to_numeric(df["outcome_time_h"], errors="coerce")
        ev = (y == 1)
        timing["events_with_time"] = sup(int((ev & ot.notna()).sum()))
        timing["events_before_index"] = sup(int((ev & (ot < 0)).sum()))
        timing["events_before_prediction_time"] = sup(int((ev & (ot < spec.prediction_time_h)).sum()))
        timing["events_before_prediction_time_frac"] = _r((ev & (ot < spec.prediction_time_h)).sum() / max(k, 1), 4)
        of = float(spec.outcome_from_h or 0.0)
        timing["events_before_outcome_from"] = sup(int((ev & (ot < of)).sum()))
        timing["events_before_outcome_from_frac"] = _r((ev & (ot < of)).sum() / max(k, 1), 4)
        if spec.exposure and spec.exposure.window_end_h is not None:
            w = spec.exposure.window_end_h
            timing["events_before_exposure_window_end"] = sup(int((ev & (ot < w)).sum()))
            timing["events_before_exposure_window_end_frac"] = _r((ev & (ot < w)).sum() / max(k, 1), 4)
        if "exposure_time_h" in df.columns:
            et = pd.to_numeric(df["exposure_time_h"], errors="coerce")
            bad = ev & et.notna() & ot.notna() & (et > ot)
            timing["exposure_after_outcome"] = sup(int(bad.sum()))
            timing["exposure_after_outcome_frac"] = _r(bad.sum() / max(k, 1), 4)
    win = spec.exposure.window_end_h if spec.exposure and spec.exposure.window_end_h is not None else spec.prediction_time_h
    if "exposure_time_h" in df.columns and spec.exposure and float(win or 0) > 0 and not spec.model.td_exposure_time_column:
        L = float(win)                                 # exposure window end; a time-dependent exposure may start later
        exposed = pd.to_numeric(df[spec.exposure.column].astype("float64") if df[spec.exposure.column].dtype == bool
                                else df[spec.exposure.column], errors="coerce") == 1
        et_all = pd.to_numeric(df["exposure_time_h"], errors="coerce")
        bad = exposed & et_all.notna() & ((et_all > L) | (et_all < 0) | ((et_all == 0) & (not spec.exposure_inclusive_at_t0)))
        timing["exposed_outside_window"] = sup(int(bad.sum()))
    if "exposure_time_h" in df.columns:
        et = pd.to_numeric(df["exposure_time_h"], errors="coerce").dropna()
        if len(et) >= K_MIN:
            timing["exposure_time_h_quartiles"] = [_r(et.quantile(q), 2) for q in (.25, .5, .75)]
    card["timing"] = timing
    # table 1
    t1 = {}
    yy = y.fillna(-1)
    vars_ = ([spec.exposure] if spec.exposure else []) + spec.covariates
    for v in vars_:
        t1[v.column] = describe_var(df[v.column], yy, v.type)
    for c in spec.model.score_columns + ([spec.model.value_column] if spec.model.value_column else []):
        if c not in t1:
            t1[c] = describe_var(df[c], yy, "numeric")
    card["table1"] = t1
    # model
    mdl = spec.model
    yc = y.where(y.isin([0, 1]))
    try:
        card["model"] = fit_main(df, spec, yc)
    except Exception as e:
        card["model"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    card["model"]["type"] = mdl.type
    # sensitivity analyses: same model refitted on rows where each flag is true
    sens = []
    for sv in spec.sensitivity:
        sub = df[_to_bool(df[sv.flag])]
        ys = yc.loc[sub.index]
        rec: dict[str, Any] = {"name": sv.name, "flag": sv.flag, "n": sup(len(sub)), "events": sup(int((ys == 1).sum()))}
        try:
            mm = fit_main(sub, spec, ys) if len(sub) >= K_MIN else {"error": "fewer than 10 units"}
            rec["primary"] = mm.get("primary") or (mm.get("logistic") or {}).get("primary")
            if mm.get("error"):
                rec["error"] = mm["error"]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        sens.append(rec)
    if sens:
        card["sensitivity"] = sens
    return card, df


def primary_value(card: dict[str, Any]) -> Optional[float]:
    """A single number tracked across iterations: OR (logistic), HR (cox), AUROC (score_eval), median (descriptive)."""
    m = card.get("model") or {}
    p = m.get("primary") or (m.get("logistic") or {}).get("primary") or {}
    for k in ("or", "hr", "auroc", "value"):
        if p.get(k) is not None:
            return float(p[k])
    return None
