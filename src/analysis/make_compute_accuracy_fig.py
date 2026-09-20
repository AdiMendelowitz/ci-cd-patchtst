"""Accuracy-vs-compute figure: test MSE against gradient updates to the best
checkpoint, CI vs CD, for two cells with substantial compute asymmetry
(leader-follower gamma=0.6 and AR(1) grid C=84/rho=0.9).

Reads directly from the two committed results CSVs already used elsewhere
in the paper; no new training runs or numbers.

Run from anywhere:

    python make_compute_accuracy_fig.py
    python make_compute_accuracy_fig.py --lf-csv path/to/lf.csv \\
        --grid-csv path/to/grid.csv --outdir path/to/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required; script only calls savefig
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

LF_CSV_NAME = "results_leader_follower.csv"
GRID_CSV_NAME = "results_grid.csv"
COLORS = {"CI": "#1f77b4", "CD": "#ff7f0e"}
PAD_EXPONENT = 0.15  # log-axis headroom as a fraction of the CD/CI ratio,
                      # tuned so the two points and their error bars clear
                      # the panel edges without excess whitespace


def _resolve_input(filename: str, explicit: str | None) -> Path:
    """Resolve an input CSV: an explicit path if given, else the nearest
    ancestor's results/<filename>, else alongside this script. Every
    candidate is anchored to this file's location, never to the caller's
    working directory, so the same invocation reads the same data
    regardless of where it is run from."""
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--input path {p} does not exist")
        return p
    here = Path(__file__).resolve().parent
    candidates = [parent / "results" / filename for parent in [here, *here.parents]]
    candidates.append(here / filename)
    for c in candidates:
        if c.exists():
            return c
    tried = ", ".join(str(c) for c in candidates[:5])
    more = " ..." if len(candidates) > 5 else ""
    raise FileNotFoundError(f"{filename} not found. Tried: {tried}{more}")


def _cell_means(df: pd.DataFrame, mask: pd.Series, source: Path) -> pd.DataFrame:
    """Mean/std test MSE and mean gradient-update count per mode, for the
    rows selected by mask. Raises with the source file and the modes
    actually present if CI or CD is missing, rather than failing later
    with an opaque KeyError on lookup."""
    sub = df[mask]
    have = set(sub["mode"].unique())
    missing = {"CI", "CD"} - have
    if missing:
        raise ValueError(
            f"{source.name}: missing mode(s) {sorted(missing)} in the "
            f"filtered cell; found {sorted(have)}")
    return sub.groupby("mode").agg(mse=("test_mse", "mean"), sd=("test_mse", "std"),
                                    U=("total_steps", "mean"))


def build_figure(lf_path: Path, grid_path: Path) -> plt.Figure:
    lf = pd.read_csv(lf_path)
    lf_s = _cell_means(lf, (lf["gamma"] == 0.6) & (lf["mode"].isin(["CI", "CD"])), lf_path)

    grid = pd.read_csv(grid_path)
    grid_mask = (grid["C"] == 84) & (grid["rho"] == 0.9) & (grid["mode"].isin(["CI", "CD"]))
    grid_s = _cell_means(grid, grid_mask, grid_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    panels = [
        (axes[0], lf_s, "Leader-follower, $\\gamma=0.6$, $C=21$"),
        (axes[1], grid_s, "AR(1) grid, $C=84$, $\\rho=0.9$"),
    ]

    for ax, s, title in panels:
        for mode in ["CI", "CD"]:
            ax.errorbar(s.loc[mode, "U"], s.loc[mode, "mse"], yerr=s.loc[mode, "sd"],
                        fmt="o", ms=8, capsize=4, color=COLORS[mode], label=mode)
        ratio = s.loc["CD", "U"] / s.loc["CI", "U"]
        ax.set_xscale("log")
        xt = [s.loc["CI", "U"], s.loc["CD", "U"]]
        ax.set_xticks(xt)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        pad = ratio ** PAD_EXPONENT
        ax.set_xlim(s.loc["CI", "U"] / pad, s.loc["CD", "U"] * pad)
        ax.set_xlabel("Gradient updates to best checkpoint")
        ax.set_title(title, fontsize=9)
        ax.annotate(f"{ratio:.1f}$\\times$", xy=(s.loc["CD", "U"], s.loc["CD", "mse"]),
                    xytext=(0, 10), textcoords="offset points", fontsize=9, ha="center")

    axes[0].set_ylabel("Test MSE")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lf-csv", default=None, help=f"Override path to {LF_CSV_NAME}.")
    parser.add_argument("--grid-csv", default=None, help=f"Override path to {GRID_CSV_NAME}.")
    parser.add_argument("--outdir", default=".", help="Directory to write the figure into.")
    args = parser.parse_args(argv)

    lf_path = _resolve_input(LF_CSV_NAME, args.lf_csv)
    grid_path = _resolve_input(GRID_CSV_NAME, args.grid_csv)
    fig = build_figure(lf_path, grid_path)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / "compute_accuracy.pdf"
    png_path = outdir / "compute_accuracy.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())