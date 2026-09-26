# SPSC: code for "Catching a Moving Subspace: Low-Rank Bandits Beyond Stationarity"

This repository reproduces every numerical table and figure of the paper. It contains:

- the SPSC algorithm, its adaptive variant, and all eleven baselines;
- the synthetic and real-data environments;
- every experiment script;
- the saved results behind the tables.

The last section maps each figure and table of the paper to its script and results file.

## Setup

Python 3.9 or later.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Versions and exact reproduction

Every script fixes its seeds. Two things can still change the numbers on another machine.

- **The four OpenML datasets.** The Pendigits, Satimage, MNIST and Fashion-MNIST environments sort the samples by class with NumPy's default sort, which is not stable. The order among samples of the same class, and so the instance, depends on the CPU's sorting kernels. Their saved results (their rows of Tables 2 and 20, Tables 13-16, and the Pendigits numbers of Table 19 and App. G.1) come from an x86 machine with AVX-512, running Linux, Python 3.11 and the versions in `requirements-linux.txt`. Other CPUs give a different instance of the same construction. The other benchmarks give the same instance on every machine.
- **SPSC's eigendecompositions.** The exact regret of an SPSC run depends on the linear-algebra library, because later actions depend on the estimated basis. Means over seeds agree within their standard errors. On the same instance, LinUCB reproduced exactly on every machine we tried.

The other results were produced with Python 3.9 and `requirements.txt`.

## Data

Every dataset downloads automatically on first use. There is nothing to fetch by hand.

| Benchmark | Source | How it is obtained |
|---|---|---|
| Covertype, Pendigits, Satimage | UCI Machine Learning Repository | `sklearn.datasets` (`fetch_covtype`, `fetch_openml`) |
| MNIST, Fashion-MNIST | OpenML | `sklearn.datasets.fetch_openml` |
| MovieLens-100K | GroupLens | downloaded from `files.grouplens.org` into `code/environments/.movielens_cache/` |
| Open Bandit (ZOZOTOWN) | Open Bandit Dataset | through the `obp` package, which ships the public sample |

The synthetic environments are generated from fixed seeds.

## Layout

```
code/
  algorithm.py            SPSC (SPSC_Algorithm1), SPSC-Adaptive (SPSC_Adaptive) and the baselines
  spsc_exact.py           Algorithm 1 of the paper, coded line by line (used by the checks of Table 19)
  probe_baselines.py      LinUCB and SW-LinUCB given the same probes as SPSC (Table 20)
  environments/           synthetic piecewise low-rank model and the real-data environments
  experiment_*.py         one script per experiment (see the map below)
  plot_phase_transition_hq.py   Figure 2
  check_aligned_coupling.py     Table 8
  make_tables.py          prints Tables 2, 3, 12-20 from the saved results
  check_window_conditioning.py  window conditioning of App. G.1
  figstyle.py             shared figure style
  results/                saved results (JSON) behind the tables
figures/                  created by the scripts
```

Run every script from inside `code/`. Scripts write results to `code/results/` and figures to `figures/`.

## Quick check from the saved results

```bash
cd code
python make_tables.py            # Tables 2, 3, 12-18, 19 and 20
python check_aligned_coupling.py # Table 8, about 2 minutes
python plot_phase_transition_hq.py   # Figure 2
python check_window_conditioning.py # App. G.1, a few minutes
```

`make_tables.py` recomputes the costed regret from each run's control regret and number of probes. It uses the fees of the paper:
- c = 0.1 on the synthetic grid;
- c = 6×10⁻⁴ on Open Bandit;
- c = 0.02 on the other real-data benchmarks.

It also prints the counts quoted in the text, such as the 51 of 58 real-data cells and the 82 of 98 cells of Table 20. A standard error can differ from the paper by one unit because of rounding.

## Reproducing each figure and table

"Saved results" are the files in `code/results/`. Rerunning a script overwrites its file. The scripts use a process pool; set `--workers` to the number of cores. The real-data grids and the synthetic grid take hours on a multi-core machine. The figure scripts for the reference setting take minutes.

| Paper | Content | Script (run in `code/`) | Saved results | Output |
|---|---|---|---|---|
| Figure 1 | Overview schematic | none (drawn in LaTeX) | none | none |
| Table 1 | Related problem classes | none | none | none |
| Figure 2 | Empirical crossover on the synthetic grid | `plot_phase_transition_hq.py` | values of Table 11 inside the script | `figures/experiment1_synthetic_phase.png` |
| Table 2 | Real-data ratios at r = 10 | `experiment_realdata_all.py`, `experiment_movielens_fitted.py`, then `make_tables.py realdata` | `experiment_realdata_all.json`, `experiment_movielens_fitted.json` | printed |
| Table 3 | SPSC-Adaptive against BOSS and Jedra | `experiment_boss_jedra_grid.py`, then `make_tables.py boss` | `experiment_boss_jedra_grid.json` | printed (also gives Algorithm 1) |
| Tables 4-7, 9, 10 | Overview, terms and rates of the bound, notation, benchmark construction, settings | none (text tables) | none | none |
| Table 8 | Subspace error under the aligned coupling | `check_aligned_coupling.py` | none (fixed seed) | printed |
| Table 11 | Synthetic grid, all 40 cells | `experiment_synthetic_extended.py` | none (printed summary) | printed rows |
| Tables 12-16 | Per-cell results: Covertype, Pendigits, Satimage, Fashion-MNIST, MNIST | `experiment_realdata_all.py`, then `make_tables.py realdata` | `experiment_realdata_all.json` | printed |
| Table 17 | Per-cell results: MovieLens (real ratings) | `experiment_movielens_fitted.py`, then `make_tables.py realdata` | `experiment_movielens_fitted.json` | printed |
| Table 18 | Per-cell results: Open Bandit (ZOZOTOWN) | `experiment_realdata_all.py --datasets openbandit`, then `make_tables.py realdata` | `experiment_realdata_all.json` | printed |
| Text of §5.2 and App. F.1 | MovieLens with probes answered by real ratings | `experiment_movielens_realprobes.py` | `experiment_movielens_realprobes.json` | JSON |
| Figure 3, App. F.3 | Small-d stress test against nonstationary baselines | `experiment9_sota_benchmark.py` | none (30 seeds, a few minutes) | `figures/experiment9_sota_benchmark.png` |
| Table 19 | Conditions of the analysis and the refresh schedule | `experiment_alg1_settings.py`, then `make_tables.py refresh` | `experiment_alg1_settings.json` | printed (also checks `spsc_exact.py` against `SPSC_Algorithm1`) |
| App. G.1 | Window conditioning in the runs | `check_window_conditioning.py` | none (fixed seeds) | printed |
| Figure 4 | Probe-rate ablation | `experiment3_probe_ablation.py` | none | `figures/experiment3_probe_ablation.png` |
| Figure 5 | Robustness and necessity | `experiment_robustness_abc.py` | none | `figures/experiment_robustness_abc.png` |
| Figure 6 | Subspace recovery within a segment | `experiment2_subspace_recovery.py` | none | `figures/experiment2_subspace_recovery_notitle.png` |
| Figure 7 | Change-point adaptation | `experiment4_changepoint_recovery.py` | none | `figures/experiment4_changepoint_recovery.png` |
| Table 20 | LinUCB and SW-LinUCB given the same probes | `experiment_sensing_control.py`, then `make_tables.py probes` | `experiment_sensing_control.json` (plus the real-data files for LinUCB and the SPSC variants) | printed |

The figure scripts can keep their raw runs between calls. Set `FIG_CACHE_DIR=/some/folder` and a second call only redraws.

## Methods in the code

| Paper | Class in `algorithm.py` |
|---|---|
| SPSC (Algorithm 1) | `SPSC_Algorithm1`. It recomputes the basis after every probe and uses the standard ridge radius in R^r, as described in App. E. `spsc_exact.py` codes Algorithm 1 line by line, and `experiment_alg1_settings.py` checks that the two agree. |
| SPSC-Adaptive | `SPSC_Adaptive`, which uses the reward-drop trigger of App. D |
| Oracle-LinUCB | `OracleLinUCB` |
| LinUCB, Restart-LinUCB | `LinUCB`, `RestartLinUCB` |
| D-LinUCB, SW-LinUCB | `LinUCB(forgetting_factor=...)`, `SWLinUCB` |
| LowOFUL, VOFUL, LowRank-Reward | `LowOFUL`, `VOFUL`, `LowRankRewardUCB` |
| LinTS, SW-LinTS | `LinTS`, `SWLinTS` |
| BOSS, Jedra | `BOSSAdapted`, `JedraAdapted` |
| Probe-LinUCB, Probe-SW-LinUCB | `ProbeLinUCB` in `probe_baselines.py` |

