"""Rewrite the manuscript around measured stage-specific relative piece values."""
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import numpy as np
from stage_analysis import ROOT, RUN, STAGES

PAPER=ROOT/'paper'
OUTPUT=ROOT.parent/'output'


def tex_table(caption, label, columns, header, rows, small=True):
    return (r'\begin{table}[htbp]\centering'+ ('\n'+r'\small' if small else '')+
            '\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+columns+'}'+
            '\n'+r'\toprule'+'\n'+header+r'\\\midrule'+'\n'+'\n'.join(rows)+
            '\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')


def row_values(est, ci=False):
    if est['ratios'] is None:return ['--']*5
    if ci:
        return [f"{v:.2f} [{est['ratio_ci95'][0][j]:.2f}, {est['ratio_ci95'][1][j]:.2f}]" for j,v in enumerate(est['ratios'])]
    return [f'{v:.2f}' for v in est['ratios']]


def main():
    s=json.loads((RUN/'summary.json').read_text())
    st=json.loads((RUN.parent/'stage_analysis'/'summary.json').read_text())
    ex=json.loads((RUN.parent/'extended_irl'/'summary.json').read_text())
    old=(PAPER/'original_main.tex').read_text(encoding='utf-8')
    preamble=old.split(r'\begin{abstract}')[0]
    title='Stage-Specific Relative Chess Piece Values\\\\Estimated with One-Step Maximum-Entropy IRL'
    start=preamble.index(r'\title{'); end=preamble.index(r'\author{')
    preamble=preamble[:start]+'\\title{'+title+'}\n'+preamble[end:]
    related=old.split(r'\section{Related work}')[1].split(r'\section{Model}')[0]
    model=old.split(r'\section{Model}')[1].split(r'\section{Experimental protocol}')[0]
    bib=r'\begin{thebibliography}'+old.split(r'\begin{thebibliography}')[1]
    main_est=st['estimates']
    names=['Pawn','Knight','Bishop','Rook','Queen']
    stage_names=['Opening','Middlegame','Endgame']
    value_rows=[names[j]+' & '+' & '.join(row_values(r,True)[j] for r in main_est)+r'\\' for j in range(5)]
    value_table=tex_table('Primary result: pooled contextual IRL coefficients normalized to pawn = 1. Brackets are pointwise 95\\% game-bootstrap intervals, conditional on fixed stage reference contexts.','tab:primary','lccc','Piece & Opening & Middlegame & Endgame',value_rows)
    raw_table=tex_table('Raw action-reward coefficients before pawn normalization. Their common numerical scale is set by temperature one and the selected regularizer.','tab:raw','lrrr','Piece & Opening & Middlegame & Endgame',[
        names[j]+' & '+' & '.join(f"{r['raw'][j]:.3f}" for r in main_est)+r'\\' for j in range(5)])
    coverage=tex_table('Training-stage reference distributions. One game can contribute to multiple stages; stage game counts must not be summed as independent games.','tab:coverage','lrrrr','Stage & Positions & Games & Mean $p$ & Mean $o$',[
        f"{name} & {r['positions']:,} & {r['games']} & {r['mean_context'][0]:.3f} & {r['mean_context'][1]:.3f}"+r'\\'
        for name,r in zip(stage_names,main_est)])
    support=tex_table('Number of training positions whose candidate moves differ in each material feature. These are opportunities to constrain that coefficient, not counts of expert captures.','tab:support','lrrrrr','Stage & Pawn & Knight & Bishop & Rook & Queen',[
        name+' & '+' & '.join(str(v) for v in r['material_variable_positions'])+r'\\' for name,r in zip(stage_names,main_est)])
    familyrows=[]
    for fam in ex['model_families']:
        for i,e in enumerate(fam['estimates']):
            familyrows.append(f"{fam['model'].title()} & {stage_names[i]} & "+' & '.join(row_values(e)[1:])+r'\\')
        familyrows.append(r'\addlinespace')
    familytable=tex_table('Alternative specifications of the same one-step maximum-entropy IRL family, evaluated at identical stage contexts. Pawn = 1. These are not four different IRL algorithms.','tab:family','llrrrr','Model & Stage & N & B & R & Q',familyrows)
    independent=ex['independent_stage_models']
    independenttable=tex_table('Independently fitted stage models: each stage has its own eight coefficients and validation-selected ridge. Brackets are 95\\% intervals from 100 bootstrap refits per stage.','tab:independent','lccc','Piece & Opening & Middlegame & Endgame',[
        names[j]+' & '+' & '.join(row_values(r['estimates'],True)[j] for r in independent)+r'\\' for j in range(5)])
    ridgerows=[]
    for r in ex['regularization']:
        for k,e in enumerate(r['estimates']):
            ridgerows.append(f"{r['ridge']:g} & {stage_names[k]} & "+' & '.join(row_values(e)[1:])+r'\\')
        ridgerows.append(r'\addlinespace')
    ridgetable=tex_table('Ridge sensitivity of the pooled joint model at the same reference contexts. No test results were used to select these settings; all three original grid values are displayed.','tab:ridge','rlrrrr',r'$\lambda$ & Stage & N & B & R & Q',ridgerows)
    sensitivity=[dict(opening_moves=10,endgame_units=6,estimates=main_est)]+st['sensitivity']
    boundaryrows=[]
    for r in sensitivity:
        for k,e in enumerate(r['estimates']):
            boundaryrows.append(f"{r['opening_moves']} / {r['endgame_units']} & {stage_names[k]} & "+' & '.join(row_values(e)[1:])+r'\\')
        boundaryrows.append(r'\addlinespace')
    boundarytable=tex_table('Sensitivity to the stage definition. Settings are opening last full move / endgame material-unit threshold. The model is frozen; only the reference populations change. Pawn = 1.','tab:boundary','llrrrr','Definition & Stage & N & B & R & Q',boundaryrows)
    diff=st['paired_contrasts'][-1]
    contrasttable=tex_table('Endgame minus opening differences in normalized coefficients of the pooled model, using paired bootstrap draws. Intervals are exploratory, pointwise, and not multiplicity-adjusted.','tab:contrast','lrr','Piece & Difference & 95\\% interval',[
        f"{names[j]} & {diff['difference'][j]:.2f} & [{diff['ci95'][0][j]:.2f}, {diff['ci95'][1][j]:.2f}]"+r'\\' for j in range(1,5)])
    metricrows=[]
    for name in ['uniform','handcrafted','global','phase','openness','joint']:
        m=s['test'][name]
        metricrows.append(f"{name.title()} & {m['game_nll']:.4f} & {100*m['game_top1']:.2f}"+r'\\')
    performance=tex_table('Supporting held-out behavioral prediction, retained as a validity diagnostic rather than the main research outcome.','tab:performance','lrr','Model & Game-mean NLL & Top-1 (\\%)',metricrows)
    stagemetrics=tex_table('Exploratory stage-level prediction diagnostics. Independent fits are evaluated only on their corresponding held-out stage. No model is selected using this table.','tab:stageperf','lrrrr','Stage & Test games & Global NLL & Joint NLL & Independent NLL',[
        f"{name} & {r['test_games']} & {r['test']['global']['game_nll']:.4f} & {r['test']['joint']['game_nll']:.4f} & {r['test']['independent']['game_nll']:.4f}"+r'\\'
        for name,r in zip(stage_names,independent)])
    qdiff=diff['ci95'][0][4],diff['ci95'][1][4]
    abstract=(f'We estimate stage-specific relative chess piece values from Stockfish 18 move choices using a one-step maximum-entropy inverse reinforcement learning model. '
              'Here, value means a model-conditional action-reward coefficient, not a uniquely identified strategic exchange price. '
              'Using 300 self-play games and 18,351 retained positions, we report opening, middlegame, and endgame coefficients normalized to pawn = 1. '
              f'In the pooled contextual model, knight estimates are {main_est[0]["ratios"][1]:.2f}, {main_est[1]["ratios"][1]:.2f}, and {main_est[2]["ratios"][1]:.2f}; '
              f'bishop estimates are {main_est[0]["ratios"][2]:.2f}, {main_est[1]["ratios"][2]:.2f}, and {main_est[2]["ratios"][2]:.2f}. '
              'We extend the analysis with independent stage fits, alternative context specifications, regularization sensitivity, and alternative stage boundaries. '
              'The results expose meaningful specification dependence and limited opening evidence for major-piece coefficients. '
              'Game-bootstrap intervals quantify conditional sampling uncertainty. All extensions are exploratory and remain within the same horizon-one IRL family; no claim of a new IRL algorithm or recovery of intrinsic piece prices is made.')
    text=preamble+r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}'+r'''

\section{Introduction}
Relative piece values summarize how a chess decision maker trades material against other features of a position. A single fixed table is convenient, but it conceals potential variation between early development, active middlegame play, and simplified endings. The purpose of this paper is to report and scrutinize IRL-estimated relative piece values at these three stages. Predictive accuracy is retained as a supporting check; the central objects are the coefficients, their uncertainty, and their dependence on modeling choices.

The term ``IRL-estimated value'' has a deliberately restricted meaning here. We infer an immediate action-reward model from demonstrated moves. We neither observe an engine's internal material weights nor recover an exact value of exchanging a piece after optimal future play. In particular, a knight coefficient below the normalized pawn coefficient must not be read as a recommendation to exchange a knight for a pawn. This distinction is essential for interpreting all tables below.

The empirical contributions are a stage-specific value table with game-bootstrap intervals; a comparison between pooled contextual estimation and independently fitted stage models; and sensitivity analyses over context features, ridge penalties, and stage boundaries. These are complementary views of the same data rather than independent replications. They reveal which conclusions depend on pooling, normalization, and available material-choice evidence. The stage analyses were developed after the original experiment and are explicitly exploratory.

\section{Related work}
'''+related+r'\section{One-step maximum-entropy IRL}'+model+r'''
\section{Data and stage definitions}
\subsection{Self-play data and partitioning}
We use the same fixed dataset throughout: 300 Stockfish 18 self-play games with one thread, a 64 MB hash and 5,000 search nodes per move\ \cite{stockfish}. Games start from standard chess, with eight uniformly sampled legal exploratory plies excluded from fitting. Subsequent moves are chosen by the engine. Every other expert position is sampled, alternating sampled color by game ID. Games stop under a rule-based termination or at 240 plies; the nine length-limited games are marked truncated, not drawn.

Games are assigned 60/20/20 to train, validation, and test using permutation seed 20260927. Forced-choice positions and exact repeated board states are removed, with training taking precedence over validation and test. The state key contains placement, turn, castling, and legal en-passant status; full histories are separately preserved. The retained splits contain 10,795, 3,624, and 3,932 positions. One training game contributes no retained positions, leaving 179, 60, and 60 represented games. This controls exact-state leakage, not all shared motifs or history-dependent effects.

\subsection{Operational stage boundaries}
Let $U=n_N+n_B+2n_R+4n_Q$ count both colors' non-pawn material. An endgame is a state with $U\leq6$, or $p\leq0.25$. This rule takes precedence. An opening is any remaining state through full move 10, equivalently pre-move ply $<20$. The first four full moves are exploratory, so observed opening examples begin at full move 5. All remaining states are middlegames. These operational boundaries do not assert a universally accepted chess taxonomy. The endgame definition can include an early liquidation, while move count alone need not identify an endgame.

\subsection{Stage summaries and uncertainty}
For the pooled contextual model, we first average the pre-move contexts within each game and stage, then average equally across contributing games. Since the raw material model is affine in context, evaluating it at this reference context equals the game-balanced mean raw coefficient. We then divide each mean coefficient by the mean pawn coefficient. This is a ratio of means, not a mean of per-position ratios.

We propagate the existing 100 training-game bootstrap coefficient draws through these fixed reference contexts. The intervals do not include uncertainty in the choice or composition of the reference population, nor hyperparameter-selection uncertainty. Paired stage differences use the same coefficient draw in both stages. All intervals are exploratory pointwise intervals, without simultaneous or multiple-comparison coverage. Ratios are withheld if the estimated pawn coefficient or any bootstrap pawn coefficient is nonpositive or near zero. A positive denominator is a numerical requirement, not a proof of substantive identification.
'''+coverage+r'''
\FloatBarrier
\section{Primary stage-specific relative values}
Table~\ref{tab:primary} is the principal result. Figure~\ref{fig:stage} visualizes its uncertainty. In this particular pooled model, minor-piece and rook coefficients decrease relative to the pawn across the three reference populations. This pattern is conditional on the feature representation and the changing positions represented in each stage. Phase and file openness are strongly correlated in training positions ($r=-0.802$), so these comparisons do not isolate causal phase effects.
'''+value_table+r'''
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{stage_values.png}
\caption{Pooled contextual IRL relative coefficients by stage. Vertical lines are pointwise 95\% intervals based on 100 game-bootstrap coefficient draws.}\label{fig:stage}
\end{figure}

\subsection{Why raw and normalized values tell different stories}
Table~\ref{tab:raw} reports the unnormalized coefficients. In particular, the pawn coefficient rises from approximately 0.636 in the opening reference population to 1.079 in the endgame population. The queen coefficient also rises, from approximately 2.911 to 3.739, even though its pawn-normalized ratio falls. Thus a decrease in a reported relative value is not evidence that the raw coefficient decreases. Normalization changes the question to value relative to the contemporaneous pawn coefficient.
'''+raw_table+contrasttable+r'''
For knight, bishop, and rook, the displayed paired opening-to-endgame difference intervals exclude zero in the pooled model. The queen difference interval includes zero. These are conditional, exploratory comparisons; they do not justify a general law that each piece loses strategic value as the game progresses.

\FloatBarrier
\section{Different IRL model specifications}
\subsection{Global, phase-only, openness-only, and joint models}
The global, phase-only, openness-only, and joint specifications have 8, 13, 13, and 18 parameters. All share the same immediate features, temperature convention, game-balanced objective, and validation procedure. Table~\ref{tab:family} evaluates them at the same stage reference distributions. By construction the global model cannot express stage variation. Phase-only and openness-only models attribute differences to distinct correlated covariates. Their disagreement is therefore informative about specification dependence rather than evidence that one covariate is a causal explanation.
'''+familytable+r'''
These are variants of one maximum-entropy conditional-choice estimator. We do not label them as separate algorithms such as Bayesian IRL, maximum-margin IRL, or adversarial IRL. No experiments with those algorithms were run. The comparison addresses how much the relative coefficients change when context is represented differently.

\subsection{Independent estimation within each stage}
As a complementary analysis, we fit a separate eight-coefficient model to each stage's training positions. We choose ridge strength from the same three-value grid using that stage's validation positions, and perform 100 training-game cluster-bootstrap refits per stage at the selected penalty. Each stage has its own nuisance coefficients for centrality, check, and mate. Unlike the pooled summary, these estimates do not borrow coefficient constraints from other stages.
'''+independenttable+r'''
The independently fitted opening estimates differ substantially from the pooled opening summary. This is a change in estimator and information sharing, not an independent replication of the same quantity. An independent model approximates the conditional choices within a stage with one fixed vector, while the pooled model uses a continuous context-dependent function learned across stages. Their different normalization denominators and nuisance coefficients can further change the ratios. Cross-stage comparisons of the independent fits should therefore be treated cautiously.

\subsection{How much direct material-choice evidence is available?}
Table~\ref{tab:support} counts positions in which at least one legal action differs in the corresponding material feature. For example, an opening position with no way to change either side's rook count does not directly distinguish rook reward weights through the material component. The opening sample contains only 13 rook-informative and 10 queen-informative positions. A wide interval or large independent opening estimate should be interpreted in light of this scarcity. The counts do not imply that all informative positions are independent observations or that the expert selected a capture.
'''+support+r'''
\FloatBarrier
\section{Sensitivity analyses}
\subsection{Regularization strength}
Table~\ref{tab:ridge} uses the weights already fitted at all three ridge settings in the original validation grid. The reference contexts are held fixed. This separates numerical regularization sensitivity from changes in stage composition. Larger ridge penalties shrink raw coefficients, but ratios need not shrink uniformly because both numerator and pawn denominator change. We report every grid setting, including those not selected by validation, rather than selecting the most intuitive value table.
'''+ridgetable+r'''
The validation-selected penalties lie on the lower boundary of the tested grid. The study does not establish optimality over penalties outside that grid. The displayed sensitivity is descriptive and does not supply confidence intervals for choosing between regularizers.

\subsection{Stage-boundary sensitivity}
We vary the last opening full move from 10 to 8 or 12, and the endgame material threshold from 6 to 4 or 8, changing one definition at a time. Table~\ref{tab:boundary} shows all settings. The fitted joint model is unchanged: these results measure changes in the reference population, not new IRL training runs. Consequently, similar numbers across nearby cutoffs demonstrate local stability of this summary procedure, not robustness to an entirely different model or dataset.
'''+boundarytable+r'''
\FloatBarrier
\section{Piece-wise interpretation}
\paragraph{Pawn.} Every normalized table sets the pawn to one by division, not by constraining its learned coefficient to be constant. The raw estimates expose its changing role in the normalization. They also show why ratios alone cannot establish whether another piece's raw contribution rose or fell.
\paragraph{Knight.} The pooled stage summaries decline strongly, but the independent opening estimate is lower than the pooled opening estimate. This discrepancy cautions against interpreting a single early-game number as a stable exchange price. In endgames, a ratio below one is a property of the fitted immediate-choice model; it does not establish that a knight is strategically inferior to a pawn.
\paragraph{Bishop.} The pooled opening ratio is relatively large and has a wide interval. It decreases in later reference populations. The context specifications and independent-stage table expose how much of this pattern depends on cross-stage pooling and on the file-openness proxy. No causal claim about opening a file or preserving a bishop pair is tested.
\paragraph{Rook.} The difference between pooled and independent opening fits is especially relevant because only a small number of opening positions can constrain rook-count preferences. Rook estimates should be reported with their uncertainty and support counts rather than rounded into a new fixed chess rule.
\paragraph{Queen.} Its raw coefficient and normalized ratio move in opposite directions across the pooled opening and endgame references. The opening-to-endgame normalized difference interval includes zero. The results therefore do not establish a consistent queen-value decrease. Opening queen evidence is particularly sparse.

\section{Limitations and conclusion}
The paper estimates model-conditional, stage-specific relative piece values within a horizon-one IRL family. Immediate feature differences cannot represent a future recapture, a mating combination several plies away, or the long-term benefit of retaining a piece. Quiet non-promoting moves usually have zero material difference. Missing continuations and omitted positional features can be absorbed into coefficients. These limitations explain why the tables must not be treated as universally valid strategic prices, although they do not quantify which omission causes each observed discrepancy.

The results come from one engine version, one node budget, random early exploration, one main game collection, and one train/validation/test split. Bootstrap intervals measure conditional sampling variation rather than generalization across engines or stronger search. Stage boundaries are operational, and phases differ in both material and pawn structure. The pooled and independent estimators have different sharing assumptions. All follow-up analyses were motivated after observing the original run and should be read as exploratory. There is no claim of a new IRL algorithm, a new discovery that piece values are contextual, or improved match strength.

The principal output is a transparent stage-specific value table together with comparisons that expose its dependence on estimation choices. Reporting pooled and independent fits, raw and normalized coefficients, and sensitivity to penalties and boundaries is more informative than presenting one numerical table as definitive. The data support further investigation with reply-aware features and independent collections, while the current results form a reproducible restricted baseline.

\section*{Data, code, and tool use}
The accompanying archive includes the full 300-game self-play data, original and extended fitting code, selected models, bootstrap coefficients, stage definitions, and machine-readable tables. The original protocol remains unchanged; an extension note records that these analyses were added later. OpenAI Codex assisted in research design, implementation, analysis, and manuscript drafting. Numerical results are computed by the supplied programs. Author identity and final author review remain required before submission; the work has not been submitted to arXiv.

\appendix
\section{Supporting predictive checks}
Behavioral prediction is not the main result, but it checks whether estimated rewards bear a measurable relationship to held-out move choices. Table~\ref{tab:performance} retains the original model comparison. The joint minus global game-mean NLL difference is -0.0057 nats, with a paired 95\% game-bootstrap interval [-0.0095, -0.0017]. The effect is small. A lower prediction loss does not establish that a coefficient is an intrinsic material value.
'''+performance+stagemetrics+r'''
\FloatBarrier
\section{Reproducibility and numerical checks}
The original analytic-gradient check had maximum absolute error $3.75\times10^{-11}$, and a synthetic conditional-choice recovery test had coefficient RMSE 0.0386. Perspective, promotion, legal candidate enumeration, board restoration, and game weighting were tested. Collected histories were replayed to verify stored states and moves. The extension additionally verifies that stage-subset target rows equal the corresponding original chosen-action feature rows and that subset game weights sum to one.

Original bootstrap coefficients are used only for pooled stage summaries. Independent-stage intervals come from 300 additional refits, 100 per stage, with stage-specific penalties fixed after validation. Stage subsetting does not move any game across the original train/validation/test partition. Neither context cutoffs nor the displayed sensitivity settings are selected using test performance. Package versions, full configurations, code hashes, and engine hashes accompany the numerical results. No fabricated value table or unexecuted algorithm is presented as an experiment.
\FloatBarrier
'''+bib
    (PAPER/'main.tex').write_text(text,encoding='utf-8')
    (PAPER/'abstract.txt').write_text(abstract,encoding='utf-8')
    shutil.copy2(RUN.parent/'stage_analysis'/'stage_values.png',PAPER/'stage_values.png')
    for _ in range(2):
        p=subprocess.run([r'C:\texlive\2026\bin\windows\pdflatex.exe','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=PAPER,capture_output=True,text=True,errors='replace')
        if p.returncode:raise RuntimeError(p.stdout[-5000:])
    shutil.copy2(PAPER/'main.pdf',OUTPUT/'pdf'/'contextual_chess_irl.pdf')
    with zipfile.ZipFile(OUTPUT/'arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','stage_values.png']:z.write(PAPER/name,name)
    # Preserve existing data and original results while adding this extension.
    with zipfile.ZipFile(OUTPUT/'reproducibility.zip') as z:entries={n:z.read(n) for n in z.namelist()}
    for name in ['stage_analysis.py','extended_irl.py','make_stage_paper.py','verify_extension.py','EXTENSION_PROTOCOL.md','README_KO.md']:
        entries['code/'+name]=(ROOT/name).read_bytes()
    for sub in ['stage_analysis','extended_irl']:
        for p in (RUN.parent/sub).iterdir():
            if p.is_file():entries['extended_results/'+sub+'/'+p.name]=p.read_bytes()
    entries['REPRODUCE_EXTENSIONS.md']=b'''# Follow-up analyses
The original experiment is unchanged. To run the extension scripts, arrange
the original fitting outputs under code/runs/main_fit (including NPZ feature
caches generated by study.py fit). Then run:
python code/stage_analysis.py
python code/extended_irl.py
The extended results are also provided directly in extended_results/.
The manuscript source archive is self-contained for LaTeX compilation.
These are exploratory specifications of one-step MaxEnt IRL, not distinct algorithms.
'''
    entries['code/paper/original_main.tex']=(PAPER/'original_main.tex').read_bytes()
    entries['manuscript/main.tex']=(PAPER/'main.tex').read_bytes()
    entries['manuscript/stage_values.png']=(PAPER/'stage_values.png').read_bytes()
    with zipfile.ZipFile(OUTPUT/'reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n,b in entries.items():z.writestr(n,b)
    print('Revised stage-focused manuscript compiled and packaged.')


if __name__=='__main__':main()
