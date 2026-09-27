# Exploratory piece-count extension

Written before executing this extension, after the earlier stage results were available. This is not a preregistration or a new independent collection.

Count all occupied squares except the two kings, including both colors' pawns. Define sparse as 0–10 non-king pieces, intermediate as 11–21, and dense as 22–30. These fixed cutoffs are not selected by validation/test performance. Preserve the original game partition and retained positions.

Fit an independent eight-feature horizon-one maximum-entropy model in each group. Select ridge from 0.0001, 0.001, 0.01 on the matching validation group; report every grid value. Estimate uncertainty with 100 paired training-game resamples drawn from the full retained training-game set, using the same counts across groups and re-normalizing the game-balanced weights within each group. Keep each group's selected penalty fixed. Report percentile intervals and paired sparse-minus-dense contrasts. Ratios require all sampled pawn coefficients to exceed 0.001; otherwise suppress normalized conclusions.

Sensitivity: refit sparse cutoffs 8 and 12 and dense cutoffs 20 and 24, selecting penalties by validation within each alternative subset. These four fits have no bootstrap intervals. Report held-out diagnostic losses, stage/group overlap, per-piece informative-position counts, and raw as well as pawn-normalized coefficients. Do not treat differences as causal effects of removing pieces. The existing group-neutral global and contextual models are comparison baselines, not newly trained IRL algorithms.
