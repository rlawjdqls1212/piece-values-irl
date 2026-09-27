"""Exploratory independently fitted piece-count IRL with paired game bootstrap."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import time
from pathlib import Path
import numpy as np
import chess
from study import Dataset, optimize, save, digest
from extended_irl import subset, normalized
from stage_analysis import labels, STAGES

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/main_fit'
OUT=ROOT/'runs/piece_count_irl'
GROUPS=['sparse','intermediate','dense']

def count_pieces(rows):
    return np.array([len(chess.Board(r['fen']).piece_map())-2 for r in rows])

def choose(tr,va):
    grid=[]
    for lam in [1e-4,1e-3,1e-2]:
        w,st=optimize(tr,list(range(8)),lam)
        metrics,_=va.evaluate(w,list(range(8)))
        grid.append(dict(ridge=lam,weights=w.tolist(),validation=metrics,optimization=st))
    best=min(grid,key=lambda r:r['validation']['game_nll'])
    return np.array(best['weights']),best['ridge'],grid

def coverage(d,stages,counts):
    info=np.array([np.ptp(d.x[o:o+n,:5],axis=0)>0 for o,n in zip(d.offsets,d.lengths)])
    return dict(positions=len(d.offsets),games=d.ngames,
                piece_count_mean=float(d.weights@counts),
                context_mean=(d.weights@d.ctx).tolist(),
                stage_positions={s:int(np.sum(stages==k)) for k,s in enumerate(STAGES)},
                informative_positions=info.sum(axis=0).tolist(),
                expert_material_change_positions=int(np.sum(np.any(d.x[d.targets,:5]!=0,axis=1))),
                mean_legal_actions=float(d.weights@d.lengths))

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True,parents=True)
    ds={}; counts={}; stages={};groups={}
    for split in ['train','validation','test']:
        z=np.load(RUN/f'{split}.npz')
        ds[split]=Dataset(**{k:z[k] for k in ['x','offsets','targets','games','ctx']})
        rows=json.loads((RUN/f'{split}_positions.json').read_text())
        assert np.array_equal(ds[split].games,[r['game_id'] for r in rows])
        counts[split]=count_pieces(rows);stages[split]=labels(rows,ds[split].ctx)
        groups[split]=np.where(counts[split]<=10,0,np.where(counts[split]>=22,2,1))
    original=json.loads((RUN/'models.json').read_text())
    records=[];training=[]
    for k,name in enumerate(GROUPS):
        sub={s:subset(ds[s],groups[s]==k) for s in ds}
        w,lam,grid=choose(sub['train'],sub['validation'])
        perf={}
        perf['independent'],_=sub['test'].evaluate(w,list(range(8)))
        for base in ['global','joint']:
            m=original[base];perf[base],_=sub['test'].evaluate(np.array(m['weights']),m['cols'])
        record=dict(group=name,weights=w.tolist(),ridge=lam,grid=grid,test=perf,
                    coverage={s:coverage(sub[s],stages[s][groups[s]==k],counts[s][groups[s]==k]) for s in ds})
        records.append(record);training.append(sub['train'])
        print(name,'fit',normalized(w),flush=True)
    # Common full-game bootstrap preserves dependence between groups of the same game.
    all_games=np.unique(ds['train'].games);rng=np.random.default_rng(2026092801)
    boots=[[] for _ in GROUPS]
    for b in range(100):
        mult=np.bincount(rng.integers(len(all_games),size=len(all_games)),minlength=len(all_games))
        for k,tr in enumerate(training):
            weights=tr.weights*mult[np.searchsorted(all_games,tr.games)]
            assert weights.sum()>0
            weights=weights/weights.sum()
            w,_=optimize(tr,list(range(8)),records[k]['ridge'],weights,np.array(records[k]['weights']))
            boots[k].append(w)
        if (b+1)%10==0:print(f'paired bootstrap {b+1}/100',flush=True)
    for k,r in enumerate(records):
        boots[k]=np.array(boots[k]);np.save(OUT/f'{GROUPS[k]}_bootstrap.npy',boots[k])
        r['estimates']=normalized(r['weights'],boots[k])
    contrast=None
    if all(r['estimates']['normalization_stable'] for r in records):
        diff=boots[0][:,:5]/boots[0][:,0,None]-boots[2][:,:5]/boots[2][:,0,None]
        contrast=dict(estimate=(np.array(records[0]['estimates']['ratios'])-np.array(records[2]['estimates']['ratios'])).tolist(),
                      ci95=np.quantile(diff,[.025,.975],axis=0).tolist())
    sensitivity=[]
    for name,cut,less in [('sparse',8,True),('sparse',12,True),('dense',20,False),('dense',24,False)]:
        masks={s:counts[s]<=cut if less else counts[s]>=cut for s in ds}
        sub={s:subset(ds[s],masks[s]) for s in ds}
        w,lam,grid=choose(sub['train'],sub['validation'])
        te,_=sub['test'].evaluate(w,list(range(8)))
        sensitivity.append(dict(group=name,cutoff=cut,ridge=lam,grid=grid,estimates=normalized(w),test=te,
                                coverage={s:coverage(sub[s],stages[s][masks[s]],counts[s][masks[s]]) for s in ds}))
        print('sensitivity',name,cut,normalized(w),flush=True)
    summary=dict(groups=records,sensitivity=sensitivity,sparse_minus_dense=contrast,bootstrap_draws=100,
                 paired=True,non_king_piece_cutoffs=[10,22],seed=2026092801,
                 code_sha256=digest(__file__),protocol_sha256=digest(ROOT/'PIECE_COUNT_PROTOCOL.md'),
                 original_models_sha256=digest(RUN/'models.json'),elapsed_seconds=time.monotonic()-start)
    save(OUT/'summary.json',summary)
    print('Completed piece-count IRL.',flush=True)

if __name__=='__main__':main()
