---
name: academic-paper-writing
description: Write and revise academic papers for ML/data-mining venues (ADMA, KDD, NeurIPS, Springer LNCS/LNAI format). Use when drafting, structuring, or polishing a conference/journal paper, its LaTeX sources, tables, or rebuttals.
---

# Academic paper writing (ML / data mining venues)

## Venue conventions
- **ADMA** (Advanced Data Mining and Applications): Springer **LNAI/LNCS**
  format (`llncs.cls`), typically 12–15 pages including references.
  Single-column, `\documentclass{llncs}`, Springer's `splncs04.bst`
  bibliography style. Not double-blind historically — check the current CFP.
- Get `llncs.cls`/`splncs04.bst` from Springer's "LaTeX template for LNCS"
  (also on CTAN and Overleaf as "Springer Lecture Notes in Computer Science").

## Structure that reviewers expect
1. **Abstract** (150–250 words): problem → gap → method (1–2 sentences per
   component) → headline results with numbers → availability of code.
2. **Introduction**: motivation, the specific limitation of prior work,
   *enumerated contributions* (3–4 bullets), each verifiable in the paper.
3. **Related work**: grouped by theme, each group ends with how this work
   differs. Never a laundry list.
4. **Method**: problem formulation first (notation table if heavy), then
   components in data-flow order, each with an equation and the *reason it
   exists*. State parameter counts and complexity.
5. **Experiments**: datasets table, protocol (look-back, horizons, splits,
   metrics), baselines with citation for every quoted number, main results
   table (bold best, underline second), ablations (one switch per variant),
   analysis/case studies, efficiency.
6. **Conclusion**: what was shown, honest limitations, future work.

## Rules of substance
- Every claim in the abstract/intro must map to a table or figure.
- Quote baseline numbers only with a source ("results from [X]") and match
  the protocol (same look-back, splits, normalization) or re-run them.
- Ablations: change exactly one component per variant; report seeds and
  note the noise floor; negative results that shaped the design are worth a
  sentence — reviewers trust papers that report them.
- Prefer "we observe X (Table N)" over adjectives. No "novel" claims for
  standard components; cite the origin (RevIN, DLinear, FITS, Mamba, etc.).
- Reproducibility paragraph: seeds, hardware, framework versions, code link.

## LaTeX practices
- Tables: `booktabs` (`\toprule/\midrule/\bottomrule`), no vertical rules;
  metrics as `MSE/MAE`; bold with `\textbf`, second-best `\underline`.
- Math: `\mathbf{X} \in \mathbb{R}^{L \times C}`; number only referenced
  equations; define every symbol at first use.
- Figures: vector (TikZ/PDF); architecture diagram in Fig. 1 near the intro.
- Citations: `\cite` clusters sorted; bib entries complete (venue, year,
  pages); use `splncs04` style for Springer.
