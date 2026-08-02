# Channel Dependence in PatchTST under Controlled Coupling

Code, data, and paper for a controlled study of channel-independent (CI) versus
channel-dependent (CD) PatchTST on multivariate time-series forecasting.

## Summary

Comparisons of CI and CD forecasting models are usually run on observational
benchmarks where correlation strength, dimensionality, and temporal dynamics all
vary together, which leaves any single CI win impossible to attribute to one
cause. This project isolates those factors one at a time using synthetic
generative processes with known structure, and treats ETTh1 and ECL as
contextual anchors rather than primary evidence.

Three synthetic families are studied: a factorial AR(1) process with
compound-symmetry covariance (instantaneous correlation only), a leader-follower
VAR(1) process with lag-1 coupling of controllable strength, and a
block-covariance family. Across every tested cell, neither mode shows a
forecast-accuracy gap that clears a pre-registered 1% relative-MSE band once each
model is selected fairly. What does not equalise is cost. The flattened-token CD
formulation needs smaller feasible batches and far more gradient steps per epoch,
its batch size collapses to one at 84 variates, and at 321 variates (ECL) it does
not finish under a single-T4 budget while CI trains stably. On accuracy per unit
compute, CI is the better default.

## Key result

On the AR(1) grid the grand-mean CD minus CI difference is +0.0013 MSE over five
seeds, with a 95% confidence interval of [-0.0002, +0.0028] and a half-width of
0.0063 MSE, so any uniform CD gain above roughly 0.63% of the CI mean would have
moved the interval off zero. None did. A fair-protocol robustness suite,
consisting of selection at each model's own validation minimum, matched
gradient-update budgets, a cross-variate prediction head, and the
block-covariance family, shows that the apparent CD deficit under lagged coupling
is largely an artefact of training protocol and selection rather than an
architectural limit.

## Repository layout

```
ci-cd-patchtst/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  paper/
    main.tex                  paper source
    references.bib
    figures/                  PNGs referenced by main.tex
  src/
    models.py                 PatchTST CI and CD modes, the PatchTST_CD_Head
                              cross-variate head, TrueDLinear, and build_model
    generators/
      generate_leader_follower.py   leader-follower VAR(1) generator
      generate_block_cov.py         block-covariance AR(1) generator
    analysis/
      analyze_synthetic.py          Table 3, Figure 1, Section 3
      analyze_realdata.py           Table 4, Table 5, Figure 2
      analyze_leader_follower.py    Table 6, Section 4.3 slope
      analyze_boundary.py           Table 7, Figure 3, Section 4.4
      analyze_overtrain.py          Figure 4, Section 5.1
      analyze_equal_compute.py      Section 5.2
      analyze_cd_head.py            Section 5.3
      analyze_block_cov.py          Section 5.4
      analyze_equiv_table.py        Table 8
      validate_granger.py           Section 2.1 Granger non-causality
      paired_stats.py               shared paired-difference library
    tests/
      test_experiments.py
      conftest.py
  notebooks/                  training notebooks (Kaggle T4); see note below
  results/                    committed result CSVs (see inventory below)
  docs/
    design_rationale.md
```

The cross-variate head is the class `PatchTST_CD_Head` inside `src/models.py`;
there is no separate `models_cd_head.py`. There is no `generate_compound.py`; the
AR(1) grid generator is not committed (see "Reproducing from scratch").

## Installation

Python 3.12 is assumed. Create a virtual environment and install with pip:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`requirements.txt` covers the analysis and plotting stack: numpy, pandas, scipy,
statsmodels, matplotlib, and seaborn. It also pins torch, which is needed only to
import `src/models.py` or to run the torch-gated model tests; the analysis scripts
that reproduce every table and figure do not import torch. All analysis runs on
CPU.

## Running the tests

```
python -m pytest src/tests/
```

The generator and paired-statistics tests run on any machine. The model tests are
skipped automatically when torch is unavailable, so a CPU-only environment without
the deep-learning stack still exercises the data-generation and statistics code. A
clean run reports passes plus skips and no failures.

## Reproducing the paper

Every table and figure is recomputed from a committed CSV in `results/` by one
analysis script. Run each from the repository root. The scripts carry the numbers
and print them on each run, so this README does not restate result values that
could drift from the paper. Figure-producing scripts write their PNG into
`paper/figures/`.

| Paper claim | Command | Reads | Prints / writes |
| --- | --- | --- | --- |
| Table 3, Figure 1, Section 3 | `python src/analysis/analyze_synthetic.py` | `results/results_grid.csv` | per-cell and grand-mean CD-CI, regression; writes `paper/figures/heatmap.png` |
| Table 4, Table 5, Figure 2 | `python src/analysis/analyze_realdata.py` | `results/results_etth1.csv`, `results/results_ecl.csv` | ETTh1 per-horizon paired CD-CI, ECL CI-only summary; writes `paper/figures/real_data.png` |
| Table 6, Section 4.3 slope | `python src/analysis/analyze_leader_follower.py` | `results/results_leader_follower.csv` | per-gamma CI, CD, and DLinear means and the gamma slope, with an oracle self-check |
| Table 7, Figure 3, Section 4.4 | `python src/analysis/analyze_boundary.py` | `results/results_boundary_p4_ci.csv`, `results/results_boundary.csv` | per-cell CD-CI across patch sizes and the boundary regression; writes `paper/figures/boundary_heatmap.png` |
| Figure 4, Section 5.1 | `python src/analysis/analyze_overtrain.py` | `results/results_overtrain_summary.csv`, `results/results_overtrain_diag.csv` | selection-rule effect sizes; writes `paper/figures/diag_overlay_clean.png` |
| Section 5.2 | `python src/analysis/analyze_equal_compute.py` | `results/results_equal_compute.csv` | matched-budget CD-CI with a validity gate on the update budget |
| Section 5.3 | `python src/analysis/analyze_cd_head.py` | `results/results_cd_head.csv` | cross-variate-head contrasts against CI and CD |
| Section 5.4 | `python src/analysis/analyze_block_cov.py` | `results/results_block_cov.csv` | block-covariance per-cell CD-CI and the rho_in slope |
| Table 8 | `python src/analysis/analyze_equiv_table.py` | `results/results_grid.csv`, `results_cd_head.csv`, `results_overtrain_summary.csv`, `results_equal_compute.csv`, `results_block_cov.csv` | recomputes all twelve equivalence rows; writes `results/equiv_summary.csv` |
| Section 2.1 Granger | `python src/analysis/validate_granger.py` | generates data internally | Granger non-causality battery (console only) |

## Results CSV inventory

`results/` holds twelve CSVs, each the committed input to one analysis script
(`equiv_summary.csv` is also written by `analyze_equiv_table.py`):

- `results_grid.csv` -- AR(1) grid (Table 3, Figure 1, Section 3)
- `results_etth1.csv`, `results_ecl.csv` -- ETTh1 and ECL (Tables 4 and 5, Figure 2)
- `results_leader_follower.csv` -- leader-follower P=16 sweep (Table 6, Section 4.3)
- `results_boundary.csv`, `results_boundary_p4_ci.csv` -- boundary patch-size sweep (Table 7, Figure 3, Section 4.4)
- `results_overtrain_summary.csv`, `results_overtrain_diag.csv` -- overtraining and selection diagnostic (Figure 4, Section 5.1)
- `results_equal_compute.csv` -- matched-compute control (Section 5.2)
- `results_cd_head.csv` -- cross-variate-head control (Section 5.3)
- `results_block_cov.csv` -- block-covariance family (Section 5.4)
- `equiv_summary.csv` -- practical-equivalence summary (Table 8)

## Notebooks

The `notebooks/` directory holds seven training notebooks, each run on a Kaggle T4
GPU, that produced the result CSVs: leader-follower, ETTh1, ECL, DLinear,
cross-variate head, equal compute, and block covariance. The AR(1) grid and the
boundary patch-size sweep are not among them; see Reproducing from scratch.

## Reproducing from scratch

Reproducing every table and figure from the committed CSVs needs only the analysis
scripts above and the CPU stack in `requirements.txt`. Retraining the models from
raw synthetic data is partially supported. The seven notebooks retrain the
experiments listed above on a single T4. The AR(1) grid and the boundary
patch-size sweep are the exception: their training notebooks (`train_grid.ipynb`,
`train_boundary.ipynb`) and the AR(1) compound-symmetry generator
(`generate_compound.py`) are not committed, so those two experiments reproduce
from their committed CSVs but cannot be retrained from this repository. The
leader-follower and block-covariance generators under `src/generators/` are
committed.

## License

Released under the MIT License; see `LICENSE`.
