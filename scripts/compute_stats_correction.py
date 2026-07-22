#!/usr/bin/env python3
"""Paired per-seed significance tests with Benjamini-Hochberg (FDR) correction.

Grounds the manuscript's statistical claims in the actual per-seed data:
  * Branch-ablation matrix (results/Branch_matrix.json): for every dataset x
    horizon cell, the frequency-branch effect (full vs time_only) and the
    time-branch effect (full vs freq_only).
  * Component ablations (results/Ablation_*.json): full vs each variant.

For each comparison we form the paired per-seed difference d_i = variant_i -
full_i (MSE), and report the mean effect, a t-based two-sided p-value
(df = n-1), and a 95% CI. p-values are then FDR-corrected with
Benjamini-Hochberg *within each family* (branch comparisons; component
ablations). Emits results/stats_correction.json and a human summary.

n = 3 seeds, so t_crit(0.975, df=2) = 4.302653.
"""
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Student-t two-sided p from a t statistic with df degrees of freedom, via the
# regularized incomplete beta function (no SciPy dependency).
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p-value P(|T| >= |t|) for Student-t with df dof."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


T_CRIT = {2: 4.302653, 3: 3.182446, 4: 2.776445, 9: 2.262157}


def paired_test(full, variant):
    """full, variant: equal-length per-seed MSE lists (paired by seed order)."""
    n = len(full)
    diffs = [v - f for v, f in zip(variant, full)]  # >0 means removing/adding hurts
    mean = sum(diffs) / n
    if n < 2:
        return dict(effect=mean, p=float("nan"), lo=mean, hi=mean, n=n)
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        t = float("inf") if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        t = mean / se
        p = t_sf_two_sided(t, df)
    tc = T_CRIT.get(df, 4.302653)
    return dict(effect=mean, p=p, lo=mean - tc * se, hi=mean + tc * se, n=n,
                t=(None if se == 0 else t))


def benjamini_hochberg(pvals):
    """Return BH q-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return q


def main():
    families = {}

    # --- Family 1: branch matrix (freq effect + time effect per cell) ---
    bm = json.load(open(RESULTS / "Branch_matrix.json"))["results"]
    branch = []
    for cell in sorted(bm):
        v = bm[cell]
        # frequency-branch effect = full vs time_only (time_only = freq removed)
        r = paired_test(v["full"], v["time_only"])
        branch.append(dict(cell=cell, branch="frequency", cmp="full_vs_time_only", **r))
        # time-branch effect = full vs freq_only (freq_only = time removed)
        r = paired_test(v["full"], v["freq_only"])
        branch.append(dict(cell=cell, branch="time", cmp="full_vs_freq_only", **r))
    families["branch"] = branch

    # --- Family 2: component ablations ---
    comp = []
    for fname, ds in [("Ablation_ETTh1.json", "ETTh1"),
                      ("Ablation_solar_final.json", "Solar"),
                      ("Ablation_weather.json", "Weather")]:
        a = json.load(open(RESULTS / fname))["results"]
        full_runs = [run["mse"] for run in a["full"]["runs"]]
        for name, entry in a.items():
            if name == "full":
                continue
            var_runs = [run["mse"] for run in entry["runs"]]
            k = min(len(full_runs), len(var_runs))
            r = paired_test(full_runs[:k], var_runs[:k])
            comp.append(dict(dataset=ds, variant=name, **r))
    families["component"] = comp

    # --- BH correction within each family ---
    out = {"seeds": 3, "t_crit_df2": T_CRIT[2], "families": {}}
    for fam, rows in families.items():
        ps = [row["p"] for row in rows]
        qs = benjamini_hochberg(ps)
        for row, q in zip(rows, qs):
            row["q_bh"] = q
        n_raw = sum(1 for row in rows if row["p"] < 0.05)
        n_bh = sum(1 for row in rows if row["q_bh"] < 0.05)
        out["families"][fam] = dict(n=len(rows), n_raw_sig=n_raw, n_bh_sig=n_bh,
                                    rows=rows)

    (RESULTS / "stats_correction.json").write_text(json.dumps(out, indent=2))

    # --- Human summary ---
    print("=" * 78)
    for fam in ("branch", "component"):
        f = out["families"][fam]
        print(f"\n### {fam.upper()} family: {f['n']} comparisons | "
              f"raw p<0.05: {f['n_raw_sig']} | BH q<0.05: {f['n_bh_sig']}")
        surv = [r for r in f["rows"] if r["q_bh"] < 0.05]
        surv.sort(key=lambda r: r["q_bh"])
        print("  Survive BH (q<0.05):")
        for r in surv:
            label = r.get("cell", "") + " " + r.get("branch", "") if fam == "branch" \
                else f"{r['dataset']}/{r['variant']}"
            print(f"    {label:34s} effect={r['effect']:+.4f} "
                  f"CI[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.4f} q={r['q_bh']:.4f}")
        # also list raw-sig-but-not-BH
        raw_only = [r for r in f["rows"] if r["p"] < 0.05 and r["q_bh"] >= 0.05]
        raw_only.sort(key=lambda r: r["p"])
        if raw_only:
            print("  Raw p<0.05 but NOT surviving BH:")
            for r in raw_only:
                label = r.get("cell", "") + " " + r.get("branch", "") if fam == "branch" \
                    else f"{r['dataset']}/{r['variant']}"
                print(f"    {label:34s} effect={r['effect']:+.4f} "
                      f"p={r['p']:.4f} q={r['q_bh']:.4f}")


if __name__ == "__main__":
    main()
