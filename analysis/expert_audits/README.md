# Expert audits

Case tables, reaction drawings and LaTeX appendix snippets for the two expert
audits in the paper. Outputs go to `results/additional/expert_audits/`.

| Audit | Appendix | Cases |
|---|---|---|
| Benchmark targets | A.2 | 53 FlowER controls, each completed by three of seven chemistry students without seeing the target |
| Equivalence metric | B.4 | 20 balanced SynRBL completions rejected by the initial metric, judged blind against the FlowER target |

## Annotation instructions

The full instructions given to the chemistry annotators are in `instructions/`:

- `curation_app_guide.pdf`: guide to the curation app used for the target audit
  (A.2) and the ongoing curation effort.
- `metric_audit_review_packet.pdf`: the blinded review packet for the metric
  audit (B.4), exactly as handed out (assessment options, then the 20 cases).

The annotators' answers:

- Target audit: all completions of the 53 three-annotator controls in
  `results/additional/expert_audits/target_audit_completions.csv`, anonymized
  as Expert 1-7. The raw app export stays local because it contains usernames.
- Metric audit: the raw blinded answers in
  `results/archive/metric_v1/additional/equivalence_metric_audit/reaction_completion_assessment_expert_first20.csv`,
  unblinded with `audit_key.csv` in the same folder.

## Rebuild

```bash
# 1. Only needed when the source data change. Reads local, non-released inputs
#    (data/Expert_curation/, written by analyze_flower_controls.py) and writes
#    anonymized case tables relabelled with the current equivalence metric.
python analysis/expert_audits/export_cases.py

# 2. Uses only the released case tables.
python analysis/expert_audits/build_appendix.py
```

Step 2 writes:

- `appendix_target_audit.tex` (manuscript, A.2): a table of the 53 controls by
  number of matching experts, one figure per agreement level (the control with
  the lowest ID: 17, 2, 177), and all three controls without a match (35, 83, 92).
  Control IDs reflect the order of the curation app, where controls were
  embedded at random.
- `appendix_metric_audit.tex` (manuscript, B.4): a table of all 20 judgements
  and one figure per expert outcome except "uncertain" (cases 4, 2, 1, 8). Each
  example is the case with the highest expert confidence for its outcome; ties
  go to the lowest case number. `SHOWN_OUTCOMES` in `build_appendix.py` sets
  which outcomes get a figure.
- `expert_audits_all_cases.tex` (repository only): figures of all 20
  metric-audit cases.
- `figures/*.png`: one image per reaction row, all at the same molecule scale.

Each manuscript snippet is a self-contained appendix section: a heading
(`\expertauditsection`, `\subsection` by default; `\renewcommand` it before the
`\input` to change the level), a label (`app:target_audit_cases`,
`app:metric_audit_cases`), a paragraph stating the selection rules, the table
with its caption above, and the figures.

Include a snippet with `\input{...}`. Images are found under `\expertauditdir`,
which defaults to `figures_camera_ready/audits` relative to the main `.tex`
file. Set another folder with `--tex-figure-dir`, or redefine the macro before
the `\input`. `--copy-figures-to paper/figures_camera_ready/audits` copies the
images the manuscript snippet uses there. Each manuscript snippet ends with `\expertauditbarrier` (default `\clearpage`),
so its figures stay in its section. With `\usepackage{placeins}`,
`\renewcommand{\expertauditbarrier}{\FloatBarrier}` avoids the page break.
The snippets need `graphicx`
and `booktabs`.

Shaded molecules differ from the FlowER target: they do not occur on the same
side of the compared reaction. Agents are drawn above the arrow and are not
compared, as in the metric. SynRBL comparisons ignore stereochemistry; target-
audit comparisons keep it.
