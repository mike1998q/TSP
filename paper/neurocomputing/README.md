# Neurocomputing submission version — DD-Mamba

Elsevier `elsarticle` manuscript (single-anonymized review → author names
ARE included at submission). Derived from the ADMA version (`../main.tex`)
with journal-specific structure and packaging.

**Positioning (post-review revision).** The manuscript is framed as a
*forecast-decomposable diagnostic framework* for **conditional component
effectiveness**, not as a novel dual-domain / time–frequency Mamba model
(that space is occupied — TF4TF, ms-Mamba, DecMamba are cited). Statistical
claims are grounded in `scripts/compute_stats_correction.py`: paired
per-seed tests with **Benjamini–Hochberg** FDR control. Under correction,
**0/29** component-placement effects survive (reported as exploratory) and
**9/54** branch-removal tests survive — 8 on the time branch (ETTm family),
and on the frequency branch **only Solar@96** (q=0.048).

**Unified same-pipeline accuracy (Table 4, `tab:unified`).** Six datasets
(ETTh1/2, ETTm1/2, Weather, Electricity) now have a five-seed, paired,
same-pipeline comparison against seven baselines
(`scripts/run_unified_baselines.py` → `results/unified_*.json`;
`scripts/analyze_unified_baselines.py` → `results/unified_analysis.json`).
DD-Mamba has the lowest average MSE on **5/6** and beats the best in-pipeline
baseline at **BH q<0.05 on 4** (Electricity, ETTh2, ETTm2, ETTh1); Weather is
a tie (q=0.23); **PatchTST is confirmed better on ETTm1**. These are the
paper's confirmatory accuracy claims. Solar/Traffic/Exchange remain
quoted-only (Table 3) pending reruns.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (`\documentclass[review,12pt]{elsarticle}`, line numbers on) |
| `refs.bib` | 28 references, `elsarticle-num` style, DOIs where verified |
| `highlights.tex` / `.pdf` | 5 highlights, each ≤85 characters (separate upload in Editorial Manager) |
| `fig_dispersion.pdf` | Training-free dispersion diagnostics (restored in the Discussion) |
| `elsarticle.cls`, `elsarticle-num.bst` | LPPL-licensed class/style (vendored so the folder compiles standalone) |
| `main.pdf` | Compiled preprint (37 pp., review format) |

Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## What differs from the ADMA version

- Elsevier frontmatter (title/author/affiliation/corresponding author),
  `\journal{Neurocomputing}`, keywords with `\sep`.
- New **Discussion** section: practical component-selection guidance (PCC
  screen), the restored dispersion-diagnostics analysis + figure, and a
  "Threats to validity" subsection (moved out of the Conclusion).
- Backmatter: CRediT authorship statement, Declaration of competing
  interest, Data availability, Acknowledgements.
- Numbered references via BibTeX (`elsarticle-num`), not inline
  `thebibliography`.
- "Camera-ready appendix" phrasing replaced (journals have no camera-ready
  appendix IOU).
- No page cap: review format runs ~30 pages (12pt, 1.5-spaced, line
  numbers); the double-column typeset version will be much shorter.

## TODO before submission (author actions)

1. Real author list, affiliations, emails, ORCIDs; corresponding author
   (`%% TODO` markers in `main.tex`). Also: author biographies, and complete
   or remove the generative-AI disclosure per the current journal policy.
2. Fill CRediT roles per real author; funding in Acknowledgements.
3. **Unified baselines — DONE for 6/9 datasets.** ETTh1/2, ETTm1/2, Weather,
   Electricity are rerun (5 seeds, one pipeline) vs DLinear, NLinear, RLinear,
   PatchTST, iTransformer, S-Mamba, ms-Mamba. **Remaining:** Solar, Traffic,
   Exchange (`python scripts/run_unified_baselines.py --config configs/solar.yaml
   --seeds 5`, etc.), then re-run `analyze_unified_baselines.py`. Optional
   hardening: swap the repo-native S-Mamba/ms-Mamba for the authors' code and
   add TF4TF via the `external_impl` adapter.
4. **Statistical strengthening:** raise seeds from 3 to ≥5 (preferably 10)
   and re-run `scripts/compute_stats_correction.py`; the current n=3 is why
   all component-placement effects are exploratory. Report raw p and BH q for
   every comparison (already emitted to `results/stats_correction.json`).
5. **Model-selection protocol:** replace the test-informed switch choices
   (e.g. Solar RevIN-off) with a fixed candidate space selected on
   validation only; report a fixed model, a validation-selected model, and a
   labeled test-oracle upper bound. Add confirmatory datasets (PEMS03/04/07/08)
   or unused temporal test blocks.
6. Optional strengtheners: complete the branch matrix on
   electricity/traffic/exchange at H=192/336/720; matched wall-clock /
   latency / throughput / peak-memory vs baselines
   (`scripts/profile_efficiency.py --device cuda`); qualitative analyses
   (branch predictions, fusion-gate distributions, frequency responses,
   failure cases); hyperparameter sensitivity study; weather PCC cell.
7. Anonymized reproducibility package **at submission** (not upon
   acceptance): data-download scripts, env/backend versions, config
   snapshots, commit ids, seed-level results, logs, table-gen scripts,
   checkpoints where feasible. Replace the repo URL `%% TODO` in `main.tex`.
8. Cover letter + 3+ suggested reviewers (Editorial Manager).
9. Check the current Guide for Authors for any changed requirements
   (graphical abstract is optional; add one from the architecture figure
   if desired).
10. Do NOT submit while the ADMA version is under review elsewhere.

## CDTF-Mamba (review-flagged citation — needs verification)

The review named **CDTF-Mamba** as a time–frequency + Mamba precedent. I
could not verify a paper under that exact name from this environment, so it
is **not** cited (no fabricated entry). Verified adjacent works ARE cited:
TF4TF (Neurocomputing 2024, time–frequency forecasting), ms-Mamba
(arXiv:2504.07654), DecMamba (CMC 2025). If CDTF-Mamba is a real reference,
add it to `refs.bib` and cite it in the "Cross-Dimensional Models" / research
gap discussion.
