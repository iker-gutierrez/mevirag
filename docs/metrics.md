# Evaluation metrics and configuration selection

MeviRAG evaluates answer quality with three complementary metrics, all
reported on a 0--100 scale:

- **ROUGE-L F1** measures lexical overlap with the reference and serves as the
  surface-form or extractive-fidelity metric.
- **BERTScore F1**, computed with `bert-base-multilingual-cased`, measures
  contextual semantic similarity and is more tolerant of paraphrases.
- **Multiple-choice accuracy (MC-acc)** measures whether the correct option is
  selected on CasiMedicos-Exp samples. It is not defined for GuiaSalud's
  open-answer samples.

The overlap metrics score the generated short answer and justification against
their corresponding references. MC-acc is computed only on the
CasiMedicos-Exp subset.

## MeanQ

MeanQ is the unweighted mean of ROUGE-L F1, BERTScore F1, and MC-acc on the
mixed GuiaSalud--CasiMedicos-Exp evaluation set. On a GuiaSalud-only table,
where MC-acc is undefined, it is the mean of ROUGE-L F1 and BERTScore F1.
MeanQ is computed independently for each seed before the mean and standard
deviation across seeds are reported.

The component metrics remain visible alongside MeanQ so that the lexical,
semantic, and decision-level behavior can be inspected separately.

## Cost metrics

The reported cost measures are:

- end-to-end wall-clock seconds per sample;
- input and generated tokens per sample, including hidden reasoning tokens and
  additional self-feedback or reasoning calls where applicable; and
- mean LLM calls per answer for the reasoning-pipeline comparison.

## MeanQ--Stability--Token selection

Development configurations are selected separately for each model with the
MeanQ--Stability--Token (MST) rule:

1. A MeanQ difference of at least 0.5 points is decisive, and the configuration
   with the higher mean wins.
2. For a smaller MeanQ difference, one point is awarded for a reduction in
   seed-to-seed standard deviation greater than 0.5 MeanQ points and one point
   for a reduction greater than 1,000 LLM tokens per answer.
3. If the stability and token-cost points remain tied, the configuration with
   the higher MeanQ wins.

All configuration selection is performed on the development split. The test
split is used only after the model-specific configuration has been frozen.
