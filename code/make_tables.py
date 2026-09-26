"""
Print the result tables of the paper from the JSON files in results/.

    python make_tables.py            # all tables
    python make_tables.py realdata   # Table 2 and the per-cell Tables 12-18
    python make_tables.py boss       # Table 3
    python make_tables.py refresh    # Table 19
    python make_tables.py probes     # Table 20

Costed regret is recomputed as control regret + fee x number of probes, with the fees of the paper:
c = 0.1 on the synthetic grid, 6e-4 on Open Bandit, and 0.02 on the other real-data benchmarks.
Every number is the mean over seeds, with the standard error where the paper reports one.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FEE = {"grid": 0.1, "openbandit": 6e-4}
DEFAULT_FEE = 0.02
NAMES = {"covertype": "Covertype", "pendigits": "Pendigits", "satimage": "Satimage", "fmnist": "Fashion-MNIST",
         "mnist": "MNIST", "movielens": "MovieLens", "openbandit": "Open Bandit"}
TABLE_NO = {"covertype": 12, "pendigits": 13, "satimage": 14, "fmnist": 15, "mnist": 16, "movielens": 17,
            "openbandit": 18}
SPSC = ("SPSC-Alg1", "SPSC-Adaptive")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def costed(x, ds):
    return x["control"] + FEE.get(ds, DEFAULT_FEE) * x["probes"]


def mean_se(v, ddof=0):
    """Mean and standard error, with the population standard deviation (ddof=0) as in every table of the paper.
    A standard error can differ from the paper by one unit from rounding."""
    v = np.asarray(v, dtype=float)
    return v.mean(), (v.std(ddof=ddof) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def realdata_rows():
    """(dataset, d, r, method) -> list of costed regrets over seeds, for the seven real-data benchmarks."""
    out = {}
    j = load("experiment_realdata_all.json")
    for x in j["rows"]:
        out.setdefault((x["dataset"], x["d"], x["r"], x["method"]), []).append(costed(x, x["dataset"]))
    methods = j["methods"]
    for x in load("experiment_movielens_fitted.json")["rows"]:
        out.setdefault(("movielens", x["d"], x["r"], x["method"]), []).append(costed(x, "movielens"))
    return out, methods


def realdata():
    rows, methods = realdata_rows()
    baselines = [m for m in methods if m not in ("Oracle-LinUCB",) + SPSC]
    cells = sorted({(ds, d, r) for (ds, d, r, _) in rows})
    summary = {}
    for ds in TABLE_NO:
        these = [(d, r) for (x, d, r) in cells if x == ds]
        if not these:
            continue
        print(f"\nTable {TABLE_NO[ds]}  {NAMES[ds]}: costed regret, mean +- SE over seeds "
              f"(* = an SPSC variant beats every non-oracle baseline, _ marks the best baseline)")
        print(f"{'d':>4} {'r':>3} " + " ".join(f"{m[:13]:>14}" for m in methods))
        for (d, r) in these:
            st = {m: mean_se(rows[(ds, d, r, m)]) for m in methods if (ds, d, r, m) in rows}
            best = min((m for m in baselines if m in st), key=lambda m: st[m][0])
            win = min(st[m][0] for m in SPSC) < st[best][0]
            txt = []
            for m in methods:
                if m not in st:
                    txt.append(f"{'--':>14}")
                    continue
                s = (f"{st[m][0]:.0f}+-{st[m][1]:.0f}" if st[m][0] >= 100 and ds != "openbandit"
                     else f"{st[m][0]:.1f}+-{st[m][1]:.1f}")
                if m == best:
                    s = "_" + s
                if m in SPSC and st[m][0] < st[best][0]:
                    s = "*" + s
                txt.append(f"{s:>14}")
            print(f"{d:>4} {r:>3} " + " ".join(txt))
            summary[(ds, d, r)] = dict(spsc=st["SPSC-Alg1"][0], adp=st["SPSC-Adaptive"][0], lin=st["LinUCB"][0],
                                       best=st[best][0], win=win)
    real = {k: v for k, v in summary.items()}
    print(f"\nAn SPSC variant beats every non-oracle baseline in {sum(v['win'] for v in real.values())} "
          f"of {len(real)} real-data cells.")
    print(f"SPSC-Adaptive has lower regret than Algorithm 1 in {sum(v['adp'] < v['spsc'] for v in real.values())} "
          f"of {len(real)} real-data cells.")
    print("\nTable 2: ratios of Algorithm 1's costed regret to LinUCB (S/Lin) and to the best non-oracle baseline "
          "(S/Best) at r = 10, and the cell with the lowest S/Lin")
    print(f"{'Dataset':14s} " + " ".join(f"{'d=' + str(d):>13}" for d in (55, 105, 200)) + "   best cell  S/Lin")
    for ds in ("covertype", "pendigits", "satimage", "mnist", "fmnist", "movielens"):
        cols = []
        for d in (55, 105, 200):
            v = summary.get((ds, d, 10))
            cols.append(f"{v['spsc'] / v['lin']:.2f} / {v['spsc'] / v['best']:.2f}" if v else "---")
        allc = {k: v for k, v in summary.items() if k[0] == ds}
        bk = min(allc, key=lambda k: allc[k]["spsc"] / allc[k]["lin"])
        print(f"{NAMES[ds]:14s} " + " ".join(f"{c:>13}" for c in cols) +
              f"   ({bk[1]},{bk[2]})  {allc[bk]['spsc'] / allc[bk]['lin']:.2f}")


def boss():
    j = load("experiment_boss_jedra_grid.json")
    print("\nTable 3: costed regret, mean +- SE over seeds, and SPSC-Adaptive against the better of BOSS and Jedra")
    print(f"{'d':>4} {'r':>3} {'SPSC-Adaptive':>15} {'BOSS':>12} {'Jedra':>12} {'vs best':>9} {'Alg. 1':>12}")
    for key, res in j["results"].items():
        d, r = [int(p.split("=")[1]) for p in key.split(",")]
        def ms(name):
            v = res[name]
            if isinstance(v, dict):
                v = v.get("costed", v.get("values", v))
            return mean_se(v)
        a, b, c, s = ms("SPSC-Adaptive"), ms("BOSS-adapted"), ms("Jedra-adapted"), ms("SPSC-Alg1")
        vs = 100 * (a[0] / min(b[0], c[0]) - 1)
        print(f"{d:>4} {r:>3} {a[0]:>9.0f}+-{a[1]:<4.0f} {b[0]:>7.0f}+-{b[1]:<3.0f} {c[0]:>7.0f}+-{c[1]:<3.0f} "
              f"{vs:>+8.1f}% {s[0]:>7.0f}+-{s[1]:<3.0f}")


def refresh():
    j = load("experiment_alg1_settings.json")
    vals = {}
    for bench, method, seed, v in j["rows"]:
        vals.setdefault((bench, method), []).append(v)
    print("\nTable 19: costed regret, mean +- SE over seeds")
    for bench in dict.fromkeys(b for b, _ in vals):
        lin = mean_se(vals[(bench, "LinUCB")])
        runs = mean_se(vals[(bench, "runs (every probe, standard radius)")])
        dbl = mean_se(vals[(bench, "runs with the 1, 2, 4, ... refresh")])
        print(f"{bench:24s} LinUCB {lin[0]:7.0f}+-{lin[1]:<4.0f} runs {runs[0]:7.0f}+-{runs[1]:<4.0f} "
              f"refresh 1,2,4,... {dbl[0]:7.0f}+-{dbl[1]:<4.0f} ({100 * (dbl[0] / runs[0] - 1):+.1f}%)")


def probes():
    """Table 20.  Probe-fed baselines against the SPSC variants.  The grid rows of LinUCB and the SPSC variants
    are in the sensing-control file; the real-data ones come from the real-data result files."""
    j = load("experiment_sensing_control.json")
    ctrl = {}
    for x in j["rows"]:
        ctrl.setdefault((x["dataset"], x["d"], x["r"], x["method"]), []).append(costed(x, x["dataset"]))
    real, _ = realdata_rows()
    def val(ds, d, r, m):
        v = ctrl.get((ds, d, r, m)) or real.get((ds, d, r, m))
        return np.mean(v) if v else None
    order = ["grid", "covertype", "pendigits", "satimage", "mnist", "fmnist", "openbandit", "movielens"]
    print("\nTable 20: medians over cells of the regret ratios, and the cells each SPSC variant wins")
    print(f"{'Benchmark':14s} {'cells':>5} {'Probe-Lin/Lin':>14} {'SPSC/Probe-Lin':>15} {'wins':>5} "
          f"{'Adp/Probe-SW':>13} {'wins':>5}")
    tot = dict(cells=0, w1=0, w2=0, either=0)
    for ds in order:
        cells = sorted({(d, r) for (x, d, r, m) in ctrl if x == ds})
        r1, r2, r3, w1, w2, either, n = [], [], [], 0, 0, 0, 0
        for (d, r) in cells:
            pl, psw = val(ds, d, r, "Probe-LinUCB"), val(ds, d, r, "Probe-SW-LinUCB-noreset")
            lin, s, a = val(ds, d, r, "LinUCB"), val(ds, d, r, "SPSC-Alg1"), val(ds, d, r, "SPSC-Adaptive")
            if None in (pl, psw, lin, s, a):
                continue
            n += 1
            r1.append(pl / lin); r2.append(s / pl); r3.append(a / psw)
            w1 += s < pl; w2 += a < psw
            either += min(s, a) < min(pl, psw)
        if not n:
            continue
        tot["cells"] += n; tot["w1"] += w1; tot["w2"] += w2; tot["either"] += either
        name = "Synthetic grid" if ds == "grid" else NAMES[ds]
        print(f"{name:14s} {n:>5} {np.median(r1):>14.2f} {np.median(r2):>15.2f} {w1:>5} {np.median(r3):>13.2f} {w2:>5}")
    print(f"{'All':14s} {tot['cells']:>5} {'':>14} {'':>15} {tot['w1']:>5} {'':>13} {tot['w2']:>5}")
    print(f"An SPSC variant beats both probe-fed baselines in {tot['either']} of {tot['cells']} cells.")


if __name__ == "__main__":
    which = sys.argv[1:] or ["realdata", "boss", "refresh", "probes"]
    for w in which:
        {"realdata": realdata, "boss": boss, "refresh": refresh, "probes": probes}[w]()
