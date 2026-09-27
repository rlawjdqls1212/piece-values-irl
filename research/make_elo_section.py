"""Generate strength-setting tables using only completed measured results."""
import json
from pathlib import Path
import numpy as np
from stage_analysis import material
from make_count_section import table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/elo_irl'

def main():
    s=json.loads((OUT/'summary.json').read_text());levels=s['levels']
    overall=[next(e for e in r['estimates'] if e['name']=='all') for r in levels]
    def cell(e,j,ci=False):
        if e['ratios'] is None:return '--'
        return f"{e['ratios'][j]:.2f}"+(f" [{e['ratio_ci95'][0][j]:.2f}, {e['ratio_ci95'][1][j]:.2f}]" if ci else '')
    vals=table(r'Joint contextual IRL at one common all-training-state reference context. Pawn-normalized coefficients with pointwise 95\% game-bootstrap intervals. Column headings are Stockfish settings, not measured human ratings.','tab:elovalues','lccc','Piece & 1500 & 2000 & 2500',[
        name+' & '+' & '.join(cell(e,j,True) for e in overall)+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen'])],True)
    stages=table('Strength-setting comparison at the same original stage reference contexts. These are pooled-model summaries, not independent cell fits. Dashes indicate failed pawn-denominator stability checks.','tab:elostages','rlrrrr','Setting & Stage & N & B & R & Q',[
        str(r['level'])+' & '+e['name'].title()+' & '+' & '.join(cell(e,j) for j in range(1,5))+r'\\' for r in levels for e in r['estimates'] if e['kind']=='stage'])
    agree=table('Agreement between saved expert moves on the same test positions, weighted equally by game.','tab:eloagree','rrr','Setting A & Setting B & Agreement',[
        f"{a['a']} & {a['b']} & {100*a['game_weighted']:.2f}"+r'\%\\' for a in s['test_label_agreement']])
    raw=table('Raw joint material coefficients at the common overall reference context.','tab:eloraw','lrrr','Piece & 1500 & 2000 & 2500',[
        name+' & '+' & '.join(f"{e['raw'][j]:.3f}" for e in overall)+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen'])])
    contrast='';interpret=''
    if overall[0]['ratios'] is not None and overall[-1]['ratios'] is not None:
        ref=overall[0]['context'];arrays=[]
        for level in [1500,2500]:
            b=np.array([material(w,ref) for w in np.load(OUT/str(level)/'joint_bootstrap.npy')]);arrays.append(b/b[:,0,None])
        ci=np.quantile(arrays[1]-arrays[0],[.025,.975],axis=0)
        delta=np.array(overall[-1]['ratios'])-np.array(overall[0]['ratios'])
        contrast=table(r'Paired setting-2500 minus setting-1500 differences at the common overall reference context; exploratory pointwise 95\% intervals.','tab:elocontrast','lrr','Piece & Difference & Interval',[
            name+f' & {delta[j]:.2f} & [{ci[0,j]:.2f}, {ci[1,j]:.2f}]'+r'\\' for j,name in enumerate(['Pawn','Knight','Bishop','Rook','Queen']) if j])
        separated=[name for j,name in enumerate(['pawn','knight','bishop','rook','queen']) if j and (ci[0,j]>0 or ci[1,j]<0)]
        interpret=('The pointwise paired intervals exclude zero for '+', '.join(separated)+'. ') if separated else 'None of the four non-pawn paired intervals excludes zero. '
        interpret+=r'Table~\ref{tab:elocontrast} reports every contrast; these intervals are not adjusted for multiple comparisons.'
        (OUT/'paired_contrasts.json').write_text(json.dumps(dict(reference=ref,estimate=delta.tolist(),ci95=ci.tolist()),indent=2))
    else:interpret='At least one overall pawn-normalization check failed; normalized between-setting contrasts are therefore withheld.'
    examples='; '.join(f"setting {r['level']}: knight {cell(e,1)}, bishop {cell(e,2)}, rook {cell(e,3)}, queen {cell(e,4)}" for r,e in zip(levels,overall))
    text=r'''\section{Stockfish Strength-Setting Experiment}
\subsection{Common-position comparison}
We extend the behavioral comparison to Stockfish 18 with \texttt{UCI\_LimitStrength=true} and \texttt{UCI\_Elo} settings 1500, 2000, and 2500. The installed engine exposes a supported range of 1320--3190. The official documentation explains that the setting targets a calibrated engine strength under specified time-control conditions \cite{stockfishuci}. Our fixed 5,000-node searches do not reproduce those calibration conditions. We therefore use the numbers strictly as engine-setting labels, not as measured Elo, human skill groups, or a verified ranking of tournament performance.

Each setting is queried on exactly the same 18,351 retained positions, with one thread, a 64 MB hash, and a fresh engine game context for every query. Boards are reconstructed from FEN; full repetition histories are not restored. The original game-level split and all legal candidate features are retained; only the selected-move labels change. This design reduces the confounding that would arise if each weakened engine generated its own different self-play state distribution. It also limits generalization: the common states were visited by the original stronger engine, and may not represent positions naturally encountered by lower-strength players.

We store every returned legal move and verify its alignment with the original FEN and game identifier. Weakened move selection can be stochastic, and the exposed engine interface does not provide a seed for reproducing that component. The archived choices are therefore the exact labels for refitting. Re-querying the same positions is a new label realization, not a promised bit-for-bit recreation. The initial proposed 1320 setting was excluded when the final experimental scope was narrowed; it is not included in any reported comparison.

\subsection{Fitting and uncertainty}
For each setting, global and joint contextual models are independently trained from its selected moves. Ridge is chosen from the same three-value grid using the matching validation labels; all selected penalties are $10^{-4}$. We then refit the selected joint model for 100 training-game bootstrap samples. The identical sequence of sampled games is used across settings, preserving pairing on the common state population. These intervals condition on the archived expert-label realization and the selected penalty. They do not measure variability from repeatedly sampling the engine's weakened move selector.

Table~\ref{tab:elovalues} summarizes the joint models at a single common context: the equal-game mean of all original training states. This keeps the reference population fixed across settings. Table~\ref{tab:eloraw} retains the raw material coefficients before dividing by the pawn term. As elsewhere, ratios are shown only when the fitted pawn term and every bootstrap denominator exceed 0.001. These checks prevent numerical division by a near-zero or sign-changing denominator, but do not establish a unique underlying reward.
'''+vals+raw+r'''
The normalized point estimates are '''+examples+r'''. These results answer how the restricted choice model responds to the different saved demonstrations on the same boards. They should not be interpreted as a table of what human players at those ratings believe a piece is worth. A weakened engine changes its move-selection procedure; that mechanism is not a model of human mistakes, learning, or judgment.

'''+interpret+'\n'+contrast+r'''
\subsection{Stage summaries and behavior diagnostics}
Table~\ref{tab:elostages} evaluates each fitted joint model at the original opening, middlegame, and endgame reference contexts. All nine cells use fixed contexts, and each setting has only one fitted joint coefficient vector. Therefore this table is a contextual summary, unlike the independently fitted count-group models. Its purpose is to reveal how the same fitted setting contrast is expressed at different state populations, not to multiply the number of independent experiments.

The opening ratios at settings 1500 and 2000 are withheld because their bootstrap pawn denominators fail the positivity threshold. These missing normalized entries reflect instability of the relative scale, not missing engine queries or an absence of pieces. The raw coefficients and all bootstrap draws remain available in the public results.
'''+stages+r'''
Table~\ref{tab:eloagree} reports agreement between the selected moves themselves, before fitting IRL. Because the states and legal alternatives are shared, disagreement directly documents a change in the available behavioral labels. Agreement includes both the systematic effect of the setting and the randomness of weakened selection in the saved realization. It cannot separate those two sources, and a low agreement rate does not quantify a difference in playing strength.
'''+agree+r'''
\subsection{What this extension adds}
The experiment changes the demonstrated actions while preserving the observed board population, complementing the count analysis that changes the fitted state subset. Together, the two extensions distinguish two sources of variation in an estimated coefficient: which decisions the demonstrator makes on a given set of boards, and which boards the estimator is asked to summarize. Neither source is an intrinsic property of the piece alone. In particular, there is no requirement that increasing the nominal setting makes every ratio converge monotonically to a conventional hand-assigned value.

The comparison remains limited by one engine family, one search budget, and one draw of weakened labels. Independent repeats would be needed to estimate engine-selector variability. A human-rating study would require human games with a specified platform, rating system, and time control. These additional designs are not substituted by the three numerical labels used here. The contribution of the present extension is a controlled common-position behavioral comparison with reproducible saved labels and model-conditional uncertainty.
'''
    (ROOT/'paper_jcst/elo_results.tex').write_text(text,encoding='utf-8')
    print('Generated measured strength-setting section.')

if __name__=='__main__':main()
