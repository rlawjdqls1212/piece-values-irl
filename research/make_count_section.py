"""Build manuscript tables directly from measured remaining-piece-count fits."""
import json
from pathlib import Path
from make_stage_paper import tex_table

ROOT=Path(__file__).resolve().parent

def table(caption,label,cols,header,rows,wide=False):
    s=tex_table(caption,label,cols,header,rows)
    if wide:s=s.replace(r'\begin{table}[htbp]',r'\begin{table*}[t]').replace(r'\end{table}',r'\end{table*}')
    return s

def main():
    s=json.loads((ROOT/'runs/piece_count_irl/summary.json').read_text());g=s['groups']
    coverage=table('Remaining-piece-count groups. Entries are positions (games). A game can contribute to several groups.','tab:countcoverage','lrrr','Group & Train & Validation & Test',[
        r['group'].title()+' & '+' & '.join(f"{r['coverage'][x]['positions']:,} ({r['coverage'][x]['games']})" for x in ['train','validation','test'])+r'\\' for r in g],True)
    values=table(r'Independently fitted piece-count models: pawn-normalized coefficients and pointwise 95\% intervals from 100 paired game-bootstrap draws.','tab:countvalues','lccc','Piece & Sparse & Intermediate & Dense',[
        name+' & '+' & '.join(f"{r['estimates']['ratios'][j]:.2f} [{r['estimates']['ratio_ci95'][0][j]:.2f}, {r['estimates']['ratio_ci95'][1][j]:.2f}]" for r in g)+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen'])],True)
    raw=table('Unnormalized coefficients for piece-count fits, including nuisance features.','tab:countraw','lrrr','Feature & Sparse & Intermediate & Dense',[
        name+' & '+' & '.join(f"{r['weights'][j]:.3f}" for r in g)+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen','Center','Check','Mate'])])
    overlap=table('Training-state overlap between count groups and stage labels. These are position counts, not independent games.','tab:countoverlap','lrrr','Group & Opening & Middle & Endgame',[
        r['group'].title()+' & '+' & '.join(str(r['coverage']['train']['stage_positions'][x]) for x in ['opening','middlegame','endgame'])+r'\\' for r in g])
    support=table('Training positions whose legal alternatives vary in each material feature.','tab:countsupport','lrrrrr','Group & P & N & B & R & Q',[
        r['group'].title()+' & '+' & '.join(str(x) for x in r['coverage']['train']['informative_positions'])+r'\\' for r in g])
    c=s['sparse_minus_dense']
    contrast=table(r'Paired sparse-minus-dense differences in normalized coefficients. Intervals are exploratory pointwise 95\% intervals.','tab:countcontrast','lrr','Piece & Difference & Interval',[
        name+f" & {c['estimate'][j]:.2f} & [{c['ci95'][0][j]:.2f}, {c['ci95'][1][j]:.2f}]"+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen']) if j])
    sensitivity=table('Refitted boundary sensitivity. These estimates have no bootstrap intervals; all thresholds were specified before the count extension was executed.','tab:countsensitivity','lrrrr','Subset & N & B & R & Q',[
        (r'$C\leq '+str(r['cutoff'])+'$' if r['group']=='sparse' else r'$C\geq '+str(r['cutoff'])+'$')+' & '+' & '.join(f'{v:.2f}' for v in r['estimates']['ratios'][1:])+r'\\' for r in s['sensitivity']])
    grid=table('All validation grid results for independently fitted count models. The penalty is selected within each group.','tab:countgrid','lrr','Group & Ridge & Validation NLL',[
        r['group'].title()+f" & {x['ridge']:g} & {x['validation']['game_nll']:.4f}"+r'\\' for r in g for x in r['grid']])
    perf=table('Held-out likelihood by remaining-piece-count group. Global and joint baselines are the original frozen models, evaluated on the same subset.','tab:countperf','lrrr','Group & Global & Joint & Independent',[
        r['group'].title()+' & '+' & '.join(f"{r['test'][x]['game_nll']:.4f}" for x in ['global','joint','independent'])+r'\\' for r in g])
    text=r'''\section{Independent IRL by Remaining-Piece Count}
\subsection{Definition and fitting design}
To examine positions with many and few pieces directly, let $C(s)$ be the number of occupied squares minus two. Both colors and all pawns are counted, while the kings are excluded. Thus the initial position has $C=30$. Sparse positions have $C\leq10$, dense positions have $C\geq22$, and the intermediate group contains $11\leq C\leq21$. These cutoffs were specified before executing this extension, after the earlier stage analysis had been observed. They were not chosen by searching for the largest contrast.

We fit a separate eight-coefficient model in each group using the original training partition. Every group has its own material and nuisance coefficients. Its ridge penalty is selected using only its corresponding validation subset. No state is assigned to a different game split, and all three count groups jointly cover every retained state. Table~\ref{tab:countcoverage} gives their sample sizes. A game can enter more than one group, so adding the group-level game counts would overstate the number of independent games.
'''+coverage+r'''
This analysis differs from summarizing the pooled model at a new reference context: every count-group coefficient is re-estimated from that group's demonstrations. It also differs from the original independent stage fits because the grouping variable counts pawns and gives every non-king piece equal weight. The phrase dense describes occupancy only; it does not assert that a position is strategically closed, tactically complicated, or difficult for an engine. Likewise, sparse positions are not assumed to be solved tablebase positions.

\subsection{Paired uncertainty and the primary comparison}
For uncertainty, we draw 100 bootstrap samples of the full set of 179 represented training games. Each draw supplies the same game multiplicity to every count group. Within a group, the original equal-game position weights are multiplied by those multiplicities and normalized again to sum to one. We refit at the selected penalty and compute pawn-normalized coefficients. This pairing preserves the dependence created when one game contributes both dense and sparse states. It does not create additional games, resample engine versions, or include uncertainty from hyperparameter selection.

Table~\ref{tab:countvalues} reports the resulting estimates. The sparse/dense knight ratios are 0.40 and 1.89, bishop ratios 0.96 and 3.29, rook ratios 1.98 and 3.39, and queen ratios 3.41 and 4.35. These are model-conditional descriptions of different state populations. The sparse knight interval crosses zero. A negative coefficient in a bootstrap draw is possible because material weights are not constrained to be positive; it is not a claim that removing a knight has a positive strategic value.
'''+values+r'''
All sampled pawn coefficients exceed the prespecified numerical threshold, so normalization is retained. Table~\ref{tab:countcontrast} reports paired sparse-minus-dense differences. The knight, bishop, and rook intervals exclude zero in this exploratory comparison. The queen interval includes zero. We therefore do not describe all four types as having a statistically established decline, and none of these intervals is adjusted for the full set of analyses in the paper.
'''+contrast+r'''
The intermediate estimates are useful because a two-group comparison alone could obscure how the omitted middle population behaves. Here the knight, bishop, and rook point estimates fall between the sparse and dense estimates. That ordering is descriptive and should not be promoted to a monotonic law over individual piece counts. Each group still contains heterogeneous material compositions, and an independent model compresses those positions into one shared vector. A model with one coefficient for every exact count would ask a different question and require stronger support for each additional parameter.

\subsection{Raw coefficients and changing denominators}
Table~\ref{tab:countraw} retains the fitted raw coefficients. The pawn coefficient is approximately 1.137 in sparse states and 0.746 in dense states. The queen coefficient moves in the opposite direction to its normalized ratio: its raw value is approximately 3.880 in sparse states and 3.247 in dense states. Consequently, the lower sparse queen-to-pawn ratio cannot by itself be described as a smaller raw queen contribution. The same distinction applies whenever normalized tables are compared across independently fitted models.
'''+raw+r'''
Nuisance coefficients are included to show that the material terms are fitted jointly with other explanations of the move. A check coefficient can be negative in a restricted group without implying that checking the opponent is intrinsically harmful. Conditional associations depend on which checks are available and selected, and omitted continuations can shift those associations. Similarly, a zero mate coefficient under ridge can result from the absence of differentiating mate opportunities. These terms are reported for transparency, not folded into material ratios or interpreted as independently identified chess principles.

\subsection{Overlap with stages and available evidence}
Table~\ref{tab:countoverlap} quantifies the overlap with the earlier stage partition. Of the 4,232 sparse training positions, 3,437 are endgames and 795 are middlegames. The dense group contains all 1,071 retained opening positions and 2,138 middlegame positions, with no endgames. The remaining 546 endgame positions belong to the intermediate count group. Thus the count analysis is related to, but is not identical to, the stage analysis. Its findings must not be counted as evidence from a new collection.
'''+overlap+r'''
The training groups also differ in their candidate actions and pawn structure. Their equal-game mean legal-action counts are approximately 18.37, 29.37, and 34.55 from sparse to dense. Mean file openness is approximately 0.912, 0.656, and 0.277. These simultaneous changes prevent a causal interpretation in which only piece count changes. Even if two positions have the same count, their available captures, promotions, checks, and quiet moves may be very different. The analysis compares those observed bundles of circumstances.

Table~\ref{tab:countsupport} reports states with at least one candidate difference in each material component. A state with no variation in a component provides no direct likelihood information for that coefficient through the material term. In the dense group, only 75 training positions distinguish queen-count alternatives, compared with 211 in the sparse group. Larger total sample size therefore does not automatically mean stronger information about every piece. These are opportunities to constrain a coefficient; they are not counts of selected captures or independent observations.
'''+support+r'''
\subsection{Penalty and boundary sensitivity}
The selected penalty is $10^{-4}$ for sparse and intermediate groups but $10^{-3}$ for the dense group. Table~\ref{tab:countgrid} shows all validation results so that this difference is visible. The dense estimate is not obtained by imposing the penalty selected for a different population. Changing regularization can alter both numerator and denominator, and the selected value need not make a ratio most similar to a conventional chess table.
'''+grid+r'''
We additionally refit sparse cutoffs of 8 and 12 and dense cutoffs of 20 and 24. These fits repeat within-subset validation selection and do not reuse a pooled coefficient without refitting. Table~\ref{tab:countsensitivity} reports every setting. The sparse knight ratio changes from approximately 0.40 at cutoff 8 to 0.83 at cutoff 12, showing that a single low sparse estimate is boundary-sensitive. The corresponding queen ratios are near 3.44 and 3.42. The dense alternatives also change the relative coefficients. These comparisons have no bootstrap intervals and cannot establish that every observed numerical difference is statistically meaningful.
'''+sensitivity+r'''
\subsection{Supporting behavior fit and interpretation}
Table~\ref{tab:countperf} evaluates the original global and joint models and the new independent model on each matching held-out subset. The joint model has slightly lower loss than the independent fit in all three groups. We do not select the main value table using this test comparison: the table's purpose is to show what an independent count-specific fit estimates. Rather, the diagnostic shows that releasing all coefficients by group does not automatically yield a predictive advantage over continuous context sharing.
'''+perf+r'''
The new analysis supports a limited conclusion. Under this feature set and these operational groups, relative minor-piece and rook coefficients differ between sparse and dense populations, while uncertainty and denominator changes complicate the queen comparison. It does not show that the same piece in the same strategic situation loses value when unrelated pieces are removed. A design aimed at that causal question would need matched or controlled position transformations, with legality and downstream tactical effects explicitly handled. Here the descriptive group comparison is the intended result.
'''
    (ROOT/'paper_jcst/count_results.tex').write_text(text,encoding='utf-8')
    print('Generated count-results section with nine measured tables.')

if __name__=='__main__':main()
