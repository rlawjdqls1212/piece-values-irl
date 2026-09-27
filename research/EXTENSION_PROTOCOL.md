# Exploratory stage-value extension

This extension was requested after the original results were observed. It is
not a preregistered confirmatory experiment. The original protocol and fitted
models are preserved.

1. Promote the frozen joint-model opening/middlegame/endgame summaries to the
   main manuscript table. Keep their fixed-context bootstrap interpretation.
2. Compare global, phase-only, openness-only and joint coefficients at exactly
   the same stage contexts. These are four specifications of the same
   horizon-one maximum-entropy IRL/conditional-logit algorithm.
3. Fit an independent eight-coefficient model within each stage, preserving
   the original game partitions. Select ridge from 1e-4, 1e-3, 1e-2 on that
   stage's validation NLL. Run 100 game-cluster bootstrap refits per stage.
   Keep stage test metrics as diagnostics; do not select the model using them.
4. Report all original joint-model ridge grid values at the same contexts.
5. Report the already computed opening 8/10/12 and endgame material-unit
   4/6/8 cutoff sensitivities. These change population summaries, not models.
6. Report raw coefficients, normalized ratios, material-feature support counts,
   normalization instability, and pointwise uncertainty. Do not label these
   runs as Bayesian, maximum-margin, adversarial, or sequential IRL.

Independent-stage bootstrap samples are generated separately by stage. Do not
use their marginal intervals as paired cross-stage difference intervals. The
original pooled contrasts use shared coefficient draws and remain explicitly
conditional on fixed reference contexts. No multiplicity correction or causal
identification is claimed.
