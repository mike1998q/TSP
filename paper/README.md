# ADMA 2026 submission — DD-Mamba

`main.tex` is a Springer LNCS/LNAI paper (ADMA proceedings format).
`main.pdf` is the compiled submission version (anonymous).

## Format compliance (checked against the official Springer LNCS
## template and the ADMA CFP)

| Requirement | Status |
|---|---|
| LNCS/LNAI format (`llncs.cls`, `[runningheads]`, T1 fonts) | ✅ compiles cleanly with the official class |
| Page limit ≤ 15 pages | ✅ 14 pages |
| Abstract 150–250 words | ✅ 250 words |
| `\keywords{...}` with `\and` separators | ✅ |
| **Double-blind review** | ✅ `\anonymoustrue` toggle: anonymous author block, no acknowledgements, code URL replaced by a "released upon acceptance" note |
| No unresolved references/citations | ✅ 0 warnings |
| No significant overfull boxes | ✅ (all < 5 pt) |

> The ADMA 2026 CFP page is not reachable from this environment's network;
> the double-blind and 15-page requirements are taken from the most recent
> ADMA CFP (2025: LNAI format, ≤ 15 pages, double-blind). Re-verify against
> the 2026 page before submitting.

## Compiling

`llncs.cls`/`splncs04.bst` come from the official Springer LNCS template
ZIP (not committed here for license reasons). Place them next to
`main.tex`, then:

```bash
pdflatex main.tex && pdflatex main.tex
```

(The bibliography is inline via `thebibliography`; no bibtex run needed.)

## Camera-ready checklist (after acceptance)

1. Set `\anonymousfalse` in `main.tex`.
2. Fill in the real author list, ORCIDs, affiliation, and email in the
   `\else` branch of the author block.
3. Fill in the acknowledgements (funding) in the `\else` branch at the end.

## Where the numbers come from

- **Our results** (Tables 3–4): 3-seed means from
  `results/Main_*.json` (RTX 5090, seq_len 96, canonical splits;
  produced by `scripts/run_main_results.py`). Solar uses the released
  RevIN-off configuration.
- **Baselines** (Table 3): quoted from the S-Mamba paper (same protocol),
  as cited in the table caption; plus a unified-framework DLinear rerun
  on ETTh1 (`results/Main_etth1_dlinear.json`).
- **Ablations** (Tables 5–6): ETTh1, Solar, and Weather, all H=96,
  3 seeds, final released configs
  (`results/Ablation_ETTh1.json`, `results/Ablation_solar_final.json`,
  `results/Ablation_weather.json`). All deltas are paired per-seed
  differences vs the full model with t-based 95% CIs (n=3,
  t=4.303) computed from the per-run values; * marks CIs excluding 0.
  The traffic RevIN probe and the traffic 4-layer-mixer negative result
  discussed in Sect. 4.3 are single-run probes.
- **Figure 1**: TikZ, self-contained in main.tex.
