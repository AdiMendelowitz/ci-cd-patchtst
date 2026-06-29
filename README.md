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
    main.pdf                  built paper
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
| Table 4, Tabl