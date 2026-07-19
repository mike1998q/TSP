# Neurocomputing submission version — DD-Mamba

Elsevier `elsarticle` manuscript (single-anonymized review → author names
ARE included at submission). Derived from the ADMA version (`../main.tex`)
with journal-specific structure and packaging.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Manuscript (`\documentclass[review,12pt]{elsarticle}`, line numbers on) |
| `refs.bib` | 28 references, `elsarticle-num` style, DOIs where verified |
| `highlights.tex` / `.pdf` | 5 highlights, each ≤85 characters (separate upload in Editorial Manager) |
| `fig_dispersion.pdf` | Training-free dispersion diagnostics (restored in the Discussion) |
| `elsarticle.cls`, `elsarticle-num.bst` | LPPL-licensed class/style (vendored so the folder compiles standalone) |
| `main.pdf` | Compiled preprint (30 pp., review format) |

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
   (`%% TODO` markers in `main.tex`).
2. Fill CRediT roles per real author; funding in Acknowledgements.
3. **Strongly recommended before submitting:** run the unified baseline
   reruns (S-Mamba/iTransformer/PatchTST in this pipeline) — at S-Mamba's
   home journal, quoted-baseline comparisons will draw fire; the manuscript
   currently states this limitation honestly.
4. Optional strengtheners: complete the branch matrix on
   electricity/traffic/exchange at H=192/336/720; wall-clock GPU
   efficiency vs baselines (`scripts/profile_efficiency.py --device cuda`);
   hyperparameter sensitivity study; weather PCC cell.
5. Cover letter + 3+ suggested reviewers (Editorial Manager).
6. Check the current Guide for Authors for any changed requirements
   (graphical abstract is optional; add one from the architecture figure
   if desired).
7. Do NOT submit while the ADMA version is under review elsewhere.
