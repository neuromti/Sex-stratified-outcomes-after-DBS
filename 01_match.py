"""
Propensity Score Matching: four independent analyses.

Q1-DBS:    F-DBS  vs M-DBS  — sex differences within DBS
Q1-BMT:    F-BMT  vs M-BMT  — sex differences within BMT
Q2-Female: DBS-F  vs BMT-F  — treatment effect in females
Q2-Male:   DBS-M  vs BMT-M  — treatment effect in males

Matching covariates: the nine PSM_COVARIATES defined in config.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

from config import *

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised Mean Difference = (mean_a - mean_b) / pooled_SD."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.nanmean(a) - np.nanmean(b)) / pooled


def build_covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return imputed covariate matrix (mean imputation for missing values)."""
    X = df[PSM_COVARIATES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Core PSM
# ─────────────────────────────────────────────────────────────────────────────

def run_psm(treated: pd.DataFrame, controls: pd.DataFrame,
            analysis_label: str) -> pd.DataFrame:
    """
    Optimal 1:1 PSM without replacement, on logit(PS), within caliper.
    treated  = DBS patients (the smaller arm)
    controls = BMT patients (the larger pool)
    Returns DataFrame of matched pairs with columns:
      subject_id_treated, subject_id_control, ps_treated, ps_control,
      + all PSM_COVARIATES for both arms.
    """
    all_df = pd.concat([treated.assign(_trt=1), controls.assign(_trt=0)],
                       ignore_index=True)
    X = build_covariate_matrix(all_df).values
    y = all_df["_trt"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(X_sc, y)
    ps = lr.predict_proba(X_sc)[:, 1]
    logit_ps = logit(ps)
    all_df["ps"]       = ps
    all_df["logit_ps"] = logit_ps

    trt_df  = all_df[all_df["_trt"] == 1].copy().reset_index(drop=True)
    ctrl_df = all_df[all_df["_trt"] == 0].copy().reset_index(drop=True)

    caliper = PSM_CALIPER * np.std(logit_ps)

    dist = np.abs(trt_df["logit_ps"].values[:, None]
                  - ctrl_df["logit_ps"].values[None, :])
    BIG  = dist.max() + 1e6                      
    cost = np.where(dist <= caliper, dist, BIG)
    row_idx, col_idx = linear_sum_assignment(cost)

    pairs = []
    for i, idx in zip(row_idx, col_idx):
        if dist[i, idx] > caliper:
            continue                          
        row      = trt_df.iloc[i]
        ctrl_row = ctrl_df.iloc[idx]
        pair = {
            "subject_id_treated": row["subject_id"],
            "subject_id_control": ctrl_row["subject_id"],
            "cohort_treated":     row.get("cohort", ""),
            "cohort_control":     ctrl_row.get("cohort", ""),
            "sex_treated":        row["sex_label"],
            "sex_control":        ctrl_row["sex_label"],
            "ps_treated":  row["ps"],
            "ps_control":  ctrl_row["ps"],
            "logit_ps_treated": row["logit_ps"],
            "logit_ps_control": ctrl_row["logit_ps"],
            "analysis": analysis_label,
        }
        for cov in PSM_COVARIATES:
            pair[f"{cov}_treated"] = row[cov] if cov in row.index else np.nan
            pair[f"{cov}_control"] = ctrl_row[cov] if cov in ctrl_row.index else np.nan
        pairs.append(pair)

    matched_df = pd.DataFrame(pairs)
    n_matched  = len(matched_df)
    n_unmatched = len(trt_df) - n_matched
    print(f"  {analysis_label}: {n_matched} matched pairs  "
          f"({n_unmatched} treated unmatched beyond caliper)")
    return matched_df


# ─────────────────────────────────────────────────────────────────────────────
# Balance diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_smd_table(treated_full: pd.DataFrame, controls_full: pd.DataFrame,
                      matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cov in PSM_COVARIATES:
        tcol = f"{cov}_treated"
        ccol = f"{cov}_control"

        a_full = pd.to_numeric(treated_full[cov], errors="coerce").dropna().values
        b_full = pd.to_numeric(controls_full[cov], errors="coerce").dropna().values
        smd_before = smd(a_full, b_full)

        if tcol in matched.columns and ccol in matched.columns:
            a_match = pd.to_numeric(matched[tcol], errors="coerce").dropna().values
            b_match = pd.to_numeric(matched[ccol], errors="coerce").dropna().values
            smd_after = smd(a_match, b_match)
        else:
            smd_after = np.nan

        rows.append({"covariate": cov,
                     "smd_before": smd_before,
                     "smd_after":  smd_after})
    return pd.DataFrame(rows)

def make_balance_table(treated: pd.DataFrame, controls: pd.DataFrame,
                       matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cov in PSM_COVARIATES:
        tcol = f"{cov}_treated"
        ccol = f"{cov}_control"
        a    = pd.to_numeric(treated[cov],   errors="coerce").dropna()
        b    = pd.to_numeric(controls[cov],  errors="coerce").dropna()
        am   = pd.to_numeric(matched[tcol],  errors="coerce").dropna() if tcol in matched else pd.Series(dtype=float)
        bm   = pd.to_numeric(matched[ccol],  errors="coerce").dropna() if ccol in matched else pd.Series(dtype=float)

        t_b, p_b = stats.ttest_ind(a, b, equal_var=False)
        t_a, p_a = stats.ttest_ind(am, bm, equal_var=False) if (len(am) > 1 and len(bm) > 1) else (np.nan, np.nan)

        rows.append({
            "covariate":     cov,
            "DBS_mean_before":  f"{a.mean():.2f}",
            "BMT_mean_before":  f"{b.mean():.2f}",
            "p_before":         f"{p_b:.3f}",
            "DBS_mean_after":   f"{am.mean():.2f}" if len(am) else "—",
            "BMT_mean_after":   f"{bm.mean():.2f}" if len(bm) else "—",
            "p_after":          f"{p_a:.3f}" if not np.isnan(p_a) else "—",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01_match.py")
    print("=" * 60)

    dbs_ppmi = pd.read_csv(OUT / "cohort_dbs_ppmi.csv")
    dbs_tue  = pd.read_csv(OUT / "cohort_dbs_tue.csv")
    bmt      = pd.read_csv(OUT / "cohort_bmt.csv")

    for df in [dbs_ppmi, dbs_tue, bmt]:
        if "sex_label" not in df.columns:
            df["sex_label"] = df["sex"].map({0.0: "F", 1.0: "M"})

    dbs_all = pd.concat([dbs_ppmi, dbs_tue], ignore_index=True)

    for cov in PSM_COVARIATES:
        for df in [dbs_all, bmt]:
            df[cov] = pd.to_numeric(df[cov], errors="coerce")
            df[cov] = df[cov].fillna(df[cov].median())

    dbs_f = dbs_all[dbs_all["sex_label"] == "F"].copy()
    dbs_m = dbs_all[dbs_all["sex_label"] == "M"].copy()
    bmt_f = bmt[bmt["sex_label"] == "F"].copy()
    bmt_m = bmt[bmt["sex_label"] == "M"].copy()

    print(f"\nDBS: F={len(dbs_f)}  M={len(dbs_m)}")
    print(f"BMT: F={len(bmt_f)}  M={len(bmt_m)}")

    # (key, treated, controls, label, treated_full, controls_full)
    # treated_full/controls_full: the full unmatched pools used for SMD-before.
    # Currently identical to treated/controls; kept separate so a future run
    # could pass a larger reference pool without changing the loop body.
    runs = [
        ("q1_dbs",    dbs_f, dbs_m, "Q1 – Sex within DBS",     dbs_f, dbs_m),
        ("q1_bmt",    bmt_f, bmt_m, "Q1 – Sex within BMT",     bmt_f, bmt_m),
        ("q2_female", dbs_f, bmt_f, "Q2 – Treatment (Female)", dbs_f, bmt_f),
        ("q2_male",   dbs_m, bmt_m, "Q2 – Treatment (Male)",   dbs_m, bmt_m),
    ]

    for key, treated, controls, label, t_full, c_full in runs:
        print(f"\n[{label}]  treated={len(treated)}  controls={len(controls)}")
        matched = run_psm(treated, controls, label)
        matched.to_csv(OUT_MATCHING / f"matched_{key}.csv", index=False)

        smd_tbl = compute_smd_table(t_full, c_full, matched)
        smd_tbl.to_csv(OUT_MATCHING_DIAG / f"smd_{key}.csv", index=False)

        bal_tbl = make_balance_table(t_full, c_full, matched)
        bal_tbl.to_csv(OUT_MATCHING_DIAG / f"balance_table_{key}.csv", index=False)


        print(smd_tbl.to_string(index=False))

    print("\n" + "=" * 60)
    print("PSM complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
