# Response to the Neurocomputing Submission-Readiness Review

This memo triages every item in the editorial review ("Neurocomputing
Submission-Readiness Review") into two groups and records what was done:

- **Adopted (revised now):** writing, consistency, and citation fixes that
  need no new compute. Each is applied in this commit; the location is cited.
- **Marked in text (cannot be revised without a GPU or author-only info):**
  items that require re-running experiments, additional seeds, official
  baseline code, or real author/repository metadata. These are documented as
  named threats in the manuscript and, where tooling already exists to run
  them, the exact script is pointed to below.

The triage principle: we only *revised* claims we could verify against the
code and result files in this repository; anything requiring evidence we do
not currently have is *disclosed* rather than silently changed.

---

## Part A — Adopted and revised in this commit

| # | Review item | What was wrong | Fix (location) |
|---|-------------|----------------|----------------|
| A1 | Time-branch raw-significance count | Manuscript said "significant in 14 cells raw" (14 is the *both-branch* total; the result file has 9 time-side + 5 freq-side). | Corrected to **9 cells raw** (8 survive BH). `main.tex` §Branch matrix. Verified against `results/stats_correction.json` (time: 9 raw / 8 BH; frequency: 5 raw / 1 BH). |
| A2 | "Frequency branch is silent at initialization" | Code (`src/models/freq_branch.py:147`) zero-inits the forecast head **only** when the FITS spectral backbone is enabled. Six of nine datasets use `backbone: none`, so their frequency head is *randomly* initialized — not silent. | Rewrote Eq. (5) surrounding text to scope the silent-init claim to the three backbone-enabled datasets (ETTh1, ETTh2, Weather) and state that elsewhere the head is randomly initialized and the RLinear-class start holds only approximately. `main.tex` §Frequency branch. |
| A3 | Test-informed language embedded in shipped configs | `configs/electricity.yaml` comments read as if the width/mixer were *selected* by beating S-Mamba on the test set ("keeps the accuracy that beats S-Mamba… d256 dropped it to a tie"), plus a leftover "confirm with one run" TODO. | Reframed comments to state the parameter-budget rationale only; moved all accuracy comparisons to the manuscript. Same cleanup applied to `configs/traffic.yaml`. |
| A4 | Config ↔ manuscript parameter-budget mismatch | Shipped configs are the *reduced* variants (Electricity `d512+shared` ≈10.6M; Traffic `d256+shared` ≈2.9M), but Table (hparams) and Table 6 numbers were measured at `d512+both` (19.7M). A reader cloning the repo would not reproduce Table 6. | Added an explicit reproducibility note to the hyperparameter-table caption: the listed widths with `mixer_placement=both` produced the reported accuracy; the default configs ship the parameter-reduced variants; set `mixer_placement=both`, `d=512` to reproduce. Config comments now point back to this caption. |
| A5 | ms-Mamba faithfulness | Paper only said the entry is a "repo-native reproduction." Our adapter realizes multi-scale by *temporally subsampling* the look-back at rates {1,2,4}, whereas published ms-Mamba uses parallel Mamba blocks with distinct per-scale state-space sampling-rate (Δ) parameters. | Added the specific architectural deviation next to the Crossformer caveat, labelling our entry a "multi-scale proxy rather than a faithful ms-Mamba." `main.tex` §Baselines. |
| A6 | Generative-AI declaration wording | Used older wording and named a tool (ChatGPT) that was not the assistant actually used. | Replaced with the current Elsevier boilerplate ("Declaration of generative AI and AI-assisted technologies in the writing process") and named the tool actually used. `main.tex` §Declaration. |
| A7 | Missing recent related work + novelty differentiation | Review noted uncited concurrent module-level frameworks (TimeRecipe, CombinationTS) and recent Mamba forecasters (TimePro), and asked us to formally differentiate our component-measurement framing. | Added and *bibliographically verified* three entries: **TimeRecipe** (Zhao et al., arXiv 2506.06482), **CombinationTS** (Wang et al., ICML 2026), and **TimePro** (Ma et al., ICML 2025, PMLR v267). TimeRecipe and CombinationTS are now the explicit closest-work comparison in the Research Gap: both benchmark module effectiveness *across* architectures, whereas we audit components *within* one forecast-decomposable model with per-seed BH-corrected tests and exploratory-labelling of test-informed switches. The text now states plainly that a formal estimand, multiple backbones, or held-out component selection would be needed to reach those frameworks' generality and that we do not claim it. `refs.bib`, `main.tex` §Related Work / §Research Gap. |
| A8 | Fusion-gate initialization bias not disclosed | The convex gate is biased so g₀≈0.9 at init, strongly favoring the linear temporal pathway; combined with the A2 init asymmetry this could depress measured branch effects. | Added a new **"Fusion-gate initialization bias"** paragraph to Threats to Validity, arguing the branch matrix should be read as a lower bound on each domain's importance under this initialization, and naming the gate-bias sweep + random-vs-zero head-init comparison that would resolve it. `main.tex` §Threats to validity. |
| A9 | Bibliography completeness (Action 8) | ms-Mamba was cited as an arXiv preprint though it is now published; the TF4TF entry was missing coauthor **Lu Zhang**. | Updated **ms-Mamba** to its published record (Neurocomputing vol. 687, art. 133646, 2026, DOI 10.1016/j.neucom.2026.133646) and added **Lu Zhang** to TF4TF's author list. `refs.bib`. |
| A10 | Positional-pairing / common-random-number caveat (Action 5) | The paper did not disclose that its paired tests use *positional* pairing, which does not guarantee a verifiable common random-number stream across architectures. | Added the caveat and the recommended unpaired/randomization sensitivity analysis to the Statistical-power threat. `main.tex` §Threats to validity. |

Citations added in A7/A9 were verified against arXiv / PMLR / the published
Neurocomputing records before inclusion; no citation was fabricated. **TSCOMP**
(named in the review alongside TimeRecipe and CombinationTS) could not be
located as a verifiable published or preprinted work from this environment, so
it was deliberately *not* cited; the authors should add it only after
confirming its bibliographic record (see Part B, B9).

---

## Part B — Marked in text (cannot be revised without a GPU or author info)

Each of these is disclosed in the manuscript (mostly in §Threats to validity
and §Discussion) rather than acted on, because doing so requires compute or
information not available in this environment. Where the tooling to execute
the item already exists in `scripts/`, it is named so the authors can run it.

| # | Review item | Why it cannot be revised now | Where disclosed / tooling |
|---|-------------|------------------------------|---------------------------|
| B1 | Validation-only switch selection (three labelled models: fixed / validation-selected / test-oracle) | Requires re-running every dataset × switch under a frozen candidate space on a GPU. | Disclosed as the "most consequential threat" (§Threats, Model-selection bias). Tooling exists: `scripts/run_selection_protocol.py` (runs the protocol) and `scripts/apply_selection.py` (writes validation-selected switches back into configs). |
| B2 | Immutable provenance / anonymized artifact deposit | Requires the authors to create the deposit and mint a DOI/URL. | §Data availability carries the "[repository URL to be inserted before submission]" placeholder. |
| B3 | Official baseline code + adding TF4TF / TimePro / Crossformer as faithful runs | Requires the authors' original code and GPU reruns through our pipeline. | §Threats (Baseline scope and tuning): accuracy claim explicitly restricted to the seven implemented baselines; Crossformer and ms-Mamba faithfulness caveats stated (A5). |
| B4 | Coverage of Solar / Traffic / Exchange in the unified table, and PEMS03/04/07/08 | New GPU reruns; PEMS are held-out confirmatory sets we have not run. | §Threats (Model-selection bias, Coverage and efficiency); §Main results restricts the ranking to the six in-pipeline datasets. |
| B5 | ≥10 seeds + falsification / robustness / efficiency (wall-clock, latency, peak memory) | More seeds and hardware measurement need a GPU. | §Threats (Statistical power and multiplicity; Coverage and efficiency): current n=3–5 results labelled exploratory where uncorrected; complexity table stated to be model-only estimates. |
| B6 | Channel-order (permutation) stress test | Requires repeated random-order GPU evaluation. | §Threats (Channel-order sensitivity). |
| B7 | Reproducing the reported budgets after config reduction | The reduced configs ship by default; verifying the reduced-vs-reported gap on Electricity/Weather/Solar needs GPU runs. | §Discussion (Complexity accounting) + hyperparameter-table caption reproducibility note (A4). |
| B8 | Author metadata (CRediT roles, acknowledgements/funding, ORCIDs, corresponding-author details, real affiliations, repository URL) | Author-only information. | `main.tex` retains `%% TODO (submission)` markers at §CRediT and §Acknowledgements; §Data availability keeps the placeholder repo URL. |
| B9 | TSCOMP citation (Action 7) | Named in the review as a component-selection framework but not locatable as a verifiable record from this environment. | Not cited (fabrication avoided); TimeRecipe and CombinationTS carry the differentiation. Authors should add TSCOMP once its record is confirmed. |
| B10 | ADMA concurrent-submission resolution (Action 8) | Requires the authors to confirm the ADMA version is withdrawn/inactive; an editorial disclosure, not a text edit. | Flagged as "Not adopted" below; must be resolved directly with the handling editor. |

Batch driver for the GPU-dependent items above: `scripts/run_revision_batch.sh`.

---

## Not adopted

- **Concurrent-submission / ADMA note.** No change is warranted from within
  the manuscript; this is an editorial disclosure the authors must make to the
  handling editor directly, not a text edit.

---

## Verification

- `pdflatex` + `bibtex` build: clean (0 undefined references/citations, 0
  overfull boxes > 10 pt).
- Numeric claims in A1 re-derived from `results/stats_correction.json`.
- A2/A4 claims re-derived from `src/models/freq_branch.py` and `configs/*.yaml`.
