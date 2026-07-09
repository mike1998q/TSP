# ADMA 2026 submission — DD-Mamba

`main.tex` is a Springer LNCS/LNAI paper (ADMA proceedings format).

## Compiling

ADMA uses the Springer LNCS class, which is not distributed with this repo.
Get `llncs.cls` (and optionally `splncs04.bst`) from the official Springer
LNCS LaTeX template:

- Springer: "LaTeX template for Lecture Notes in Computer Science" —
  https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
- Or on Overleaf: search "Springer Lecture Notes in Computer Science".

Place `llncs.cls` next to `main.tex`, then:

```bash
pdflatex main.tex
pdflatex main.tex   # second pass for references/labels
```

(The bibliography is inline via `thebibliography`, so no bibtex run is
needed.)

## Before submission — TODOs in main.tex

1. Replace the placeholder author list, affiliation, and email.
2. Fill in the acknowledgements (funding).
3. Check the ADMA 2026 CFP for the current page limit (historically 12–15
   pages LNCS) and whether the review is double-blind — if so, anonymize
   the author block and the code URL.
4. Optional: add an architecture figure as Fig. 1 (the README diagram in
   the repo root is the reference layout).

## Where the numbers come from

- **Our results** (Tables 2–3): the per-horizon runs reported in this
  project (RTX 5090, seq_len 96, canonical splits) — the same numbers as
  the repo README's Results section.
- **Baselines** (Table 2): quoted from the S-Mamba paper (same protocol),
  as cited in the table caption.
- **Ablations** (Table 4): `scripts/run_ablation.py` outputs on
  Solar-Energy (H=96), plus the traffic RevIN probe and the traffic
  4-layer-mixer negative result discussed in Sect. 4.3.
