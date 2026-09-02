"""
Post-treatment outcome levels, 6-36 months after T0, for the four matched cohorts.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import *

warnings.filterwarnings("ignore")

# ── Options ───────────────────────────────────────────────────────────────────

COMPLETE_PAIRS_ONLY = "--complete-pairs" in sys.argv
ADJUST_FOR_BASELINE = "--no-baseline-adjust" not in sys.argv
SUFFIX = ("_completepairs" if COMPLETE_PAIRS_ONLY else "") + \
         ("" if ADJUST_FOR_BASELINE else "_unadj")

COMMON_WINDOW = (6, 36)

# An outcome is fitted only if both arms reach this many participants.
MIN_PER_ARM = 8

# Only MoCA is scored so that a higher value is the better outcome.
HIGHER_IS_BETTER = {"moca"}

T0_COV_TO_OUTCOME = {
    "updrs2_T0":     "updrs2",
    "updrs3_on_T0":  "updrs3_on",
    "updrs4_T0":     "updrs4",
    "moca_T0":       "moca",
    "ledd_T0":       "ledd",
}

# The two MDS-UPDRS Part I halves are analysed separately
EXTRA_T0_TO_OUTCOME = {
    "updrs1_rater_T0":   "updrs1_rater",
    "updrs1_patient_T0": "updrs1_patient",
}


# ── Assembling one matched cohort ─────────────────────────────────────────────

def load_cohort_baselines() -> pd.DataFrame:
    """Participant-indexed table of the T0 values held only in the cohort files."""
    frames = []
    for name in ["cohort_dbs_ppmi", "cohort_dbs_tue", "cohort_bmt"]:
        df = pd.read_csv(OUT / f"{name}.csv")
        frames.append(df[["subject_id"] + [c for c in EXTRA_T0_TO_OUTCOME
                                           if c in df.columns]])
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("subject_id").set_index("subject_id"))


def build_cohort(key: str, group_col: str, treated_label: str,
                 control_label: str, long: pd.DataFrame) -> pd.DataFrame:
    """
    Post-T0 visits for one matched cohort, carrying arm membership, the matched
    pair identifier, and every outcome's baseline value.
    """
    matched = pd.read_csv(OUT_MATCHING / f"matched_{key}.csv")

    arm, pair, base = {}, {}, {}
    for pair_id, (_, row) in enumerate(matched.iterrows()):
        for side, label in [("treated", treated_label), ("control", control_label)]:
            sid = row[f"subject_id_{side}"]
            arm[sid], pair[sid] = label, pair_id
            base[sid] = {outcome: pd.to_numeric(row.get(f"{cov}_{side}"),
                                                errors="coerce")
                         for cov, outcome in T0_COV_TO_OUTCOME.items()}

    df = long[long["subject_id"].isin(arm) & (long["months_since_T0"] > 0)].copy()
    df[group_col] = df["subject_id"].map(arm)
    df["pair_id"] = df["subject_id"].map(pair)
    df["months"]  = df["months_since_T0"]

    for outcome in T0_COV_TO_OUTCOME.values():
        df[f"{outcome}_base"] = df["subject_id"].map(lambda s: base[s][outcome])

    extra = load_cohort_baselines()
    for col, outcome in EXTRA_T0_TO_OUTCOME.items():
        if col in extra.columns:
            df[f"{outcome}_base"] = pd.to_numeric(df["subject_id"].map(extra[col]),
                                                  errors="coerce")
    return df


# ── One outcome, one cohort ───────────────────────────────────────────────────

def fit_window(df: pd.DataFrame, outcome: str, group_col: str,
               treated_label: str, control_label: str) -> dict:
    """Adjusted difference in level (treated minus control) inside the window."""
    lo, hi   = COMMON_WINDOW
    base_col = f"{outcome}_base"
    has_base = (ADJUST_FOR_BASELINE and base_col in df.columns
                and df[base_col].notna().any())

    w = df[(df["months"] >= lo) & (df["months"] < hi)].copy()
    w[outcome] = pd.to_numeric(w[outcome], errors="coerce")
    w = w.dropna(subset=[outcome] + ([base_col] if has_base else []))

    row = {"outcome": outcome, "label": OUTCOMES.get(outcome, outcome),
           "window": f"{lo}-{hi}m", "baseline_adj": has_base}
    if w.empty:
        row["error"] = "no data in window"
        return row

    # One row per participant: the mean of their visits inside the window, and
    # the mean timing of exactly those visits.
    agg = {outcome: "mean", "months": "mean",
           group_col: "first", "pair_id": "first"}
    if has_base:
        agg[base_col] = "first"
    w = w.groupby("subject_id").agg(agg).reset_index()

    if COMPLETE_PAIRS_ONLY:
        both = w.groupby("pair_id").size()
        w = w[w["pair_id"].isin(both[both == 2].index)]

    w["group_bin"] = (w[group_col] == treated_label).astype(float)
    t, c = w[w["group_bin"] == 1], w[w["group_bin"] == 0]

    row.update({
        "n_treated": len(t), "n_control": len(c),
        "mean_treated": t[outcome].mean(), "mean_control": c[outcome].mean(),
        "exposure_treated": t["months"].mean(),
        "exposure_control": c["months"].mean(),
        "n_pairs_repr": w["pair_id"].nunique(),
        "n_complete_pairs": int((w.groupby("pair_id").size() == 2).sum()),
        "n_singleton_pairs": int((w.groupby("pair_id").size() == 1).sum()),
    })
    row["raw_diff"] = row["mean_treated"] - row["mean_control"]

    if len(t) < MIN_PER_ARM or len(c) < MIN_PER_ARM:
        row["error"] = "too few participants in window"
        return row

    formula = f"{outcome} ~ group_bin + months" + (f" + {base_col}" if has_base else "")
    res = smf.ols(formula, data=w).fit(cov_type="cluster",
                                       cov_kwds={"groups": w["pair_id"]})
    ci = res.conf_int().loc["group_bin"]
    row.update({
        "adj_diff": res.params["group_bin"],
        "se":       res.bse["group_bin"],
        "p":        res.pvalues["group_bin"],
        "ci_lo": float(ci.iloc[0]), "ci_hi": float(ci.iloc[1]),
    })

    sd = w[outcome].std(ddof=1)
    row["std_effect"] = row["adj_diff"] / sd if sd else np.nan

    better_when_negative = outcome not in HIGHER_IS_BETTER
    row["favours"] = (treated_label
                      if (row["adj_diff"] < 0) == better_when_negative
                      else control_label)
    return row


def benjamini_hochberg(res: pd.DataFrame) -> pd.DataFrame:
    """Step-up BH q-values across the outcomes of one cohort."""
    res["q"] = np.nan
    fam = res[res["p"].notna()]
    if fam.empty:
        return res

    p = fam["p"].values
    n = len(p)
    q = np.empty(n)
    running = 1.0
    for rank, idx in enumerate(reversed(np.argsort(p)), start=1):
        running = min(running, p[idx] * n / (n - rank + 1))
        q[idx] = running

    res.loc[fam.index, "q"] = q
    return res


# ── Main ──────────────────────────────────────────────────────────────────────

ANALYSES = [
    ("q1_dbs",    "sex_label", "F",   "M",   "Q1-DBS    sex within DBS"),
    ("q1_bmt",    "sex_label", "F",   "M",   "Q1-BMT    sex within medical management"),
    ("q2_female", "group",     "DBS", "BMT", "Q2-Female treatment in women"),
    ("q2_male",   "group",     "DBS", "BMT", "Q2-Male   treatment in men"),
]

def main():
    lo, hi = COMMON_WINDOW
    print(f"Post-treatment outcome levels, {lo}-{hi} months after T0")
    if COMPLETE_PAIRS_ONLY:
        print("  sensitivity analysis: complete pairs only")
    if not ADJUST_FOR_BASELINE:
        print("  sensitivity analysis: no baseline adjustment")

    long = pd.read_csv(OUT / "longitudinal.csv", low_memory=False)
    long["months_since_T0"] = pd.to_numeric(long["months_since_T0"], errors="coerce")

    summary = []
    for key, group_col, treated, control, label in ANALYSES:
        df = build_cohort(key, group_col, treated, control, long)
        res = pd.DataFrame([fit_window(df, o, group_col, treated, control)
                            for o in OUTCOMES if o in df.columns])
        res = benjamini_hochberg(res)
        res.insert(0, "cohort", key)
        res.to_csv(OUT_LEVELS / f"results_{key}{SUFFIX}.csv", index=False)
        summary.append(res)

        print(f"\n[{label}]   estimate is {treated} minus {control}")
        print(f"  {'outcome':16s}{'diff':>9}{'95% CI':>20}{'p':>9}{'q':>9}"
              f"{'n':>10}{'pairs':>14}")
        for _, r in res.iterrows():
            if pd.isna(r.get("adj_diff")):
                print(f"  {r['outcome']:16s}{'not estimable':>9}")
                continue
            ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            n  = f"{int(r['n_treated'])}/{int(r['n_control'])}"
            pr = f"{int(r['n_complete_pairs'])}+{int(r['n_singleton_pairs'])}"
            print(f"  {r['outcome']:16s}{r['adj_diff']:>9.2f}{ci:>20}"
                  f"{r['p']:>9.4f}{r['q']:>9.4f}{'*' if r['q'] < .05 else ' '}"
                  f"{n:>9}{pr:>14}")
        print(f"  pairs column: complete + singleton clusters "
              f"(of {df['pair_id'].nunique()} matched pairs)")

    pd.concat(summary, ignore_index=True).to_csv(
        OUT_LEVELS / f"results_summary{SUFFIX}.csv", index=False)
    print(f"\nSaved to {OUT_LEVELS}")


if __name__ == "__main__":
    main()
