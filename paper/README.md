# ADMA 2026 submission — DD-Mamba

`main.tex` is a Springer LNCS/LNAI paper (ADMA proceedings format).
`main.pdf` is the compiled submission version (anonymous).

> **Current version.** `main.tex` is the conservative revision: it drops the
> dispersion head, the gaps schematic, and the gate/diagnostics figures, and
> reframes explicit dispersion forecasting as future work. It uses **two
> self-contained TikZ figures** (architecture; Mamba/mixer block internals)
> and needs **no external image files** to compile. The dispersion-head code,
> diagnostics, and efficiency tests remain in the repo for reproducibility but
> are not part of this version of the paper; `fig_evidence.pdf` /
> `fig_dispersion.pdf` are likewise unused by it.

## Format compliance (checked against the official Springer LNCS
## template and the ADMA CFP)

| Requirement | Status |
|---|---|
| LNCS/LNAI format (`llncs.cls`, `[runningheads]`, T1 fonts) | ✅ compiles cleanly with the official class |
| Page limit ≤ 15 pages | ✅ 15 pages (at the limit — no headroom) |
| Abstract 150–250 words | ✅ 250 words |
| `\keywords{...}` with `\and` separators | ✅ |
| **Double-blind review** | ✅ `\anonymoustrue` toggle: anonymous author block, no acknowledgements, code URL replaced by a "released upon acceptance" note |
| No unresolved references/citations | ✅ 0 warnings (28 references, alphabetized) |
| No significant overfull boxes | ✅ (all < 5 pt) |

> The ADMA 2026 CFP page is not reachable from this environment's network;
> the double-blind and 15-page requirements are taken from the most recent
> ADMA CFP (2025: LNAI format, ≤ 15 pages, double-blind). Re-verify against
> the 2026 page before submitting.

## Branch-ablation matrix (run — now in the paper)

The resolving experiment for the frequency-branch question has been run on
the RTX 5090 and is Table 6 of the paper: full / time-only / freq-only,
three paired seeds, on all nine datasets at H=96 AND across all four
horizons (96/192/336/720) on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Solar — 27
cells total (`results/Branch_matrix.json`, produced by
`scripts/run_branch_matrix.py`). Findings: frequency branch significant in
5/27 cells, concentrated at H=96 (Solar/ETTm1/Electricity/Exchange, plus
ETTm2@336) and NOT persisting to longer horizons on Solar; time branch
significant in 9/27 cells, including ETTm1 at every horizon; no branch
effect on ETTh1/ETTh2/Weather at any horizon or Traffic@96. Every
full-model mean reproduces the per-horizon main table, and the Solar@96
cell replicates the earlier ablation (+0.0126 vs +0.0114, overlapping
CIs). Remaining gap: Electricity/Traffic/Exchange at H=192/336/720
(resumable: `python scripts/run_branch_matrix.py --datasets electricity
traffic exchange_rate --horizons 192 336 720`).

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
- **Ablations** (Tables 5 and 7): ETTh1, Solar, and Weather, all H=96,
  3 seeds, final released configs
  (`results/Ablation_ETTh1.json`, `results/Ablation_solar_final.json`,
  `results/Ablation_weather.json`). All deltas are paired per-seed
  differences vs the full model with t-based 95% CIs (n=3,
  t=4.303) computed from the per-run values; * marks CIs excluding 0.
  The traffic RevIN probe and the traffic 4-layer-mixer negative result
  discussed in Sect. 4.3 are single-run probes.
- **Branch matrix** (Table 6): `results/Branch_matrix.json` — 27 cells (9
  datasets at H=96; 6 datasets across all 4 horizons), 3 paired seeds, from
  `scripts/run_branch_matrix.py` (RTX 5090).
- **Table 8** (complexity profile): params and MACs/forecast are
  hardware-independent; **peak GPU memory and throughput are measured on the
  device** (the model runs on GPU). Measured by `scripts/profile_efficiency.py`,
  which auto-selects CUDA:
  ```bash
  python scripts/profile_efficiency.py --device cuda --batch 32   # full GPU profile
  python scripts/profile_efficiency.py --dispersion --device cuda # head overhead on GPU
  ```
  Timing uses warm-up + `torch.cuda.synchronize()`; peak memory uses
  `torch.cuda.max_memory_allocated`. On a CPU-only host it falls back with a
  warning and reports `nan` peak memory (GPU figures require CUDA).
  - **Dispersion-head efficiency (params, hardware-independent):** the head
    adds a fixed **13,056 parameters** regardless of channel count (shared
    across variates), GMACs unchanged to three decimals — 14% of the tiny
    Exchange model but only **0.07% on Traffic**.
  - Pytest: `test_dispersion_head_efficiency` asserts the param overhead
    (== head size, channel-independent, `<10%` of the base model);
    `test_dispersion_head_gpu_efficiency` measures peak GPU memory and
    throughput on CUDA (skipped automatically when no GPU is present).
- **Figure 1**: colour-coded, layered architecture diagram — teal
  normalization, blue time branch and orange frequency branch in shaded
  background panels, violet convex-gate fusion (TikZ, self-contained).
  Needs the `fit` and `backgrounds` TikZ libraries (already in the preamble).
- **Figure 2**: module-internals diagram (R2-style) of the two
  selective-state-space blocks — (a) the causal Mamba layer (selective scan,
  conv, SiLU gate, residual) and (b) the bidirectional variate mixer
  (forward/backward scans + FFN) — TikZ, self-contained.
Not used by this version of the paper (kept in the repo for reproducibility):
`fig_dispersion.pdf` (`scripts/dispersion_diagnostics.py`, needs
`statsmodels`) and `fig_evidence.pdf` (`scripts/make_evidence_figure.py`).
The paper compiles with no external image files.

The reference list is 28 entries (Transformer, linear/MLP, frequency, SSM,
and decomposition families), alphabetized.

## Reference papers

`R1.pdf` (Dudek, STD, TKDE 2023) and `R2.pdf` (Gao et al., DFIR-DETR,
Neural Networks 2026) informed earlier revisions; the current version cites
STD (`std`) in the decomposition-and-heteroscedasticity paragraph and no
longer cites DFIR-DETR. They are not part of the submission and should be
removed before packaging the camera-ready.
