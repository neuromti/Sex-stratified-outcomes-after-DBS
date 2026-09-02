# Sex-stratified outcomes in matched Parkinson's disease cohorts

Analysis code for:

> **Propensity-matched sex-stratified outcomes after deep brain stimulation in Parkinson's disease**
> Frieder Wizgall, Farzin Negahbani, Idil Cebi, Daniel Weiss, Alireza Gharabaghi
> Institute for Neuromodulation and Neurotechnology, University Hospital and University of Tübingen

Four independently propensity-matched cohorts, aligned on nine baseline variables,
compare female with male participants within deep brain stimulation (DBS) and within
medical management (MM), and DBS with MM within each sex. Eight outcomes are assessed
6–36 months after the reference time point T0.

| Contrast | Comparison | Matched pairs |
|---|---|---:|
| Q1-DBS | female vs male, both after DBS | 59 |
| Q1-MM | female vs male, both on medical management | 103 |
| Q2-Female | DBS vs medical management, women only | 38 |
| Q2-Male | DBS vs medical management, men only | 72 |

## Contents

| File | Purpose |
|---|---|
| `config.py` | Data paths, matching covariates, outcomes, analysis constants |
| `01_match.py` | Propensity models, optimal 1:1 matching, balance diagnostics |
| `02_analyse.py` | Outcome models over the 6–36 month window, Benjamini–Hochberg correction |

```
python3 01_match.py
python3 02_analyse.py                        # reported analysis
python3 02_analyse.py --complete-pairs       # sensitivity: complete pairs only
python3 02_analyse.py --no-baseline-adjust   # sensitivity: no baseline term
```

Python 3.12.7 with pandas 2.2.2, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.7.0,
statsmodels 0.14.2.

## Data

The scripts document the analysis and cannot be executed without the two source
datasets, neither of which is redistributable.

- **PPMI** (Parkinson's Progression Markers Initiative), downloaded 20 April 2026,
  available to qualified researchers at
  [ppmi-info.org](https://www.ppmi-info.org/access-data-specimens/download-data)
  (RRID:SCR_006431), subject to PPMI data-use agreements.
- **Tübingen clinical cohort** — participant-level data cannot be made publicly
  available because of institutional ethics, data-protection and patient-privacy
  restrictions.

Cohort assembly from the raw sources is performed upstream and is not included here.
The scripts expect participant-level cohort tables (`cohort_dbs_ppmi.csv`,
`cohort_dbs_tue.csv`, `cohort_bmt.csv`) and a visit-level table (`longitudinal.csv`)
in `outputs/`.

## Method

Optimal 1:1 matching without replacement on the logit propensity score, caliper
0.3 SD. Each participant contributes one value per outcome — the mean of their
assessments in the window — modelled as `Y ~ group + months + Y_T0` by ordinary
least squares with cluster-robust standard errors clustered on the matched pair.
Multiplicity is controlled by Benjamini–Hochberg within each cohort across its
eight outcomes.
