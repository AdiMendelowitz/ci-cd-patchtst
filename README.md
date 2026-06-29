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
    main.tex                 paper source
    references.bib
    figures/                 figures referenced by main.tex
  src/
    models.py                PatchTST (CI and CD modes)
    models_cd_head.py        cross-variate prediction head variant
    generators/              generate_compound.py, generate_leader_follower.py,
                             generate_block_cov.py
    analysis/                analyze_synthetic.py, analyze_realdata.py,
                             analyze_boundary.py, analyze_cd_head.py,
                             analyze_equal_compute.py, analyze_block_cov.py,
                             validate_granger.py, paired_stats.py
    tests/                   test_experiments.py
  notebooks/                 train_grid, train_leader_follower, train_etth1,
                             train_ecl, train_boundary, train_dlinear,
                             train_cd_head, train_equal_compute,
                             train_block_cov  (all .ipynb)
  results/                   canonical CSVs only
  docs/
    design_rationale.md
```

## Installation

Python 3.12 is assumed. With uv:

```
uv venv
uv pip install -r requirements.txt
```

GPU training was run on single NVIDIA T4 notebooks; the analysis and plotting
scripts run on CPU.

## Reproducing the experiments

1. Generate the synthetic data with the scripts in `src/generators/`. Real
   datasets (ETTh1, ECL) are downloaded separately; synthetic data is
   regenerated from seeds and is not tracked.
2. Train with the notebooks in `notebooks/`. Each writes a canonical CSV to
   `results/`.
3. Analyse with the scripts in `src/analysis/`. Each reads its canonical CSV and
   reproduces the reported statistics and figures.
4. Build the paper from `paper/main.tex`.

Every number in the paper reproduces from a canonical CSV in `results/` before it
is written, so the analysis scripts are the single source of truth. Paired
statistics (per-seed differences, confidence intervals, and the regression
helpers) are shared through `src/analysis/paired_stats.py`.

## Results

`results/` holds the canonical CSVs only. The main ones are the AR(1) grid
(`results_grid.csv`), the leader-follower sweep
(`results_leader_follower.csv`), the ETTh1 matched-budget runs
(`results_etth1.csv`), the ECL CI-only runs (`results_ecl.csv`),
the boundary sweep, and the three robustness CSVs for the cross-variate head,
matched-compute, and block-covariance controls.

## Citing

```bibtex
@misc{mendelowitz2026cicd,
  author       = {Adi Mendelowitz},
  title        = {Equal Accuracy, Unequal Cost: Channel Dependence in
                  PatchTST under Controlled Coupling},
  year         = {2026},
  howpublished = {\url{https://github.com/AdiMendelowitz/ci-cd-patchtst}}
}
```

Replace the URL with the final repository location, and add a Zenodo DOI here if
you archive a release.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The paper text and
figures are released under CC-BY-4.0.

## Contact

Adi Mendelowitz, adimendelowitz@gmail.com.
GitHub: github.com/AdiMendelowitz. Site: adimendelowitz.dev.
