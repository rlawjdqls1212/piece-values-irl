# Standard chess contextual one-step maximum-entropy IRL

Protocol written before new data collection. Prior variant-chess documents are superseded for this project.

## Scope

Estimate interpretable, context-dependent action reward coefficients from Stockfish 18 move choices. This is a horizon-one/contextual-bandit inverse reward model, mathematically conditional logistic regression. It does not recover unique sequential chess rewards or intrinsic piece prices. No evaluation scores are used as training labels. No claim of a novel IRL algorithm.

## Data

Stockfish 18, one thread, 64 MB hash, fixed node budget, standard starting position. Each game begins with 8 uniformly sampled legal exploratory plies, excluded from training. All subsequent engine moves are recorded through natural termination or 240 plies. Truncation is not a draw. Clear engine state between games. Seeds uniquely identify games; preserve PGN and full UCI history. Use every second expert position to reduce adjacent-state redundancy, alternating the sampled color by game ID. Positions with one legal action are recorded but excluded from estimation and evaluation because they convey no choice information.

Pilot: 12 games, seed 700000, 1,000 nodes/move. Pilot is for correctness and runtime only; not pooled into the paper dataset. Initial main dataset: 300 games, seeds 10000--10299, 5,000 nodes/move. A larger run can be added if resources permit; report any change. Deterministic train/validation/test allocation 60/20/20 at game level (split seed 20260927), followed by removal of repeated board states within a split and across splits with train then validation then test precedence. Key includes pieces, turn, castling and legal en-passant square; history is retained separately. This prevents identical boards from inflating predictive evaluation but does not guarantee independent strategic motifs. State coverage and class counts are reported.

## Features and context

Eight immediate feature differences, from the player-to-move's fixed perspective: pawn, knight, bishop, rook, queen count differences; signed per-piece centrality summed over the board; delivered check; delivered mate. Terminal draws have no special learned bonus. State-only terms cancel across candidate actions. Thus material coefficients are informed predominantly by captures/promotions, and cannot alone explain saving or repositioning a piece.

Phase p = min(1, (N+B+2R+4Q)/24), counting both colors. p=1 is full non-pawn material; p=0 is a pawn ending. File openness o = fraction of files missing at least one color's pawn (semi-open or fully open). It is a structural proxy, not a complete definition of open versus closed chess. Context is computed before the candidate move, centered at 0.5. Context modifies the five material coefficients only; the three nuisance coefficients are shared.

Models: uniform legal policy; hand-set 8-vector [1,3,3.2,5,9,0.6,0.5,15] with inverse temperature fitted on train; learned global 8-vector; phase-only (13 parameters); openness-only (13); joint phase+openness (18). Same feature definitions and data for all models. Softmax temperature is 1 for learned models. Ridge lambda from {1e-4,1e-3,1e-2}, selected using validation game-mean NLL. Training is game-balanced maximum likelihood plus lambda/2 times squared coefficient norm. Freeze models before test evaluation.

## Evaluation and uncertainty

Primary: held-out game-mean negative log likelihood. Secondary: position-mean NLL and tie-adjusted top-1 accuracy; report uniform baseline. Main contrast: joint versus global. Paired game bootstrap (2,000 draws) for test NLL difference. Phase-only/openness-only contrasts are exploratory.

Coefficient uncertainty: training-game cluster bootstrap with selected lambda fixed, at least 100 refits if feasible. Report actual successful count. Reference contexts are empirical train joint-context tercile cells with at least 20 training games contributing; report actual p,o and coverage. Display raw coefficients as well as pawn-normalized ratios. Suppress ratios when the bootstrap 95% interval for pawn crosses zero or has nonpositive lower bound. Ratios are not identified universal prices; report sensitivity to regularization.

No test-driven hyperparameter tuning, no removal of poor outcomes, no fabricated wins or value estimates. No win-rate claim without separate controlled matches. State whether results are exploratory and list all deviations from this protocol.

## Verification

Analytic gradients versus finite differences, perspective and promotion/capture fixtures, game split leakage checks, synthetic contextual coefficient recovery, PGN replay and engine legality checks, deterministic resume and data hashes. Record package versions, engine hash, raw data hashes, configuration and timings.
