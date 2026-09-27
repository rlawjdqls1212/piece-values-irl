"""Exploratory IRL specification comparisons, preserving the original run."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import time
import numpy as np
from study import Dataset, optimize, save, digest
from stage_analysis import ROOT, RUN, STAGES, labels, game_mean, material

OUT=ROOT/'runs'/'extended_irl'


def subset(d, mask):
    ids=np.flatnonzero(mask)
    blocks=[d.x[d.offsets[i]:d.offsets[i]+d.lengths[i]] for i in ids]
    offsets=np.r_[0,np.cumsum([len(x) for x in blocks])[:-1]]
    targets=offsets+d.targets[ids]-d.offsets[ids]
    sub=Dataset(np.vstack(blocks),offsets,targets,d.games[ids],d.ctx[ids])
    assert np.array_equal(sub.x[sub.targets],d.x[d.targets[ids]])
    assert np.isclose(sub.weights.sum(),1)
    return sub


def normalized(w, boots=None):
    raw=np.asarray(w)[:5]
    out=dict(raw=raw.tolist(), ratios=(raw/raw[0]).tolist() if raw[0]>1e-3 else None)
    if boots is not None:
        bs=np.asarray(boots)[:,:5]
        out['raw_ci95']=np.quantile(bs,[.025,.975],axis=0).tolist()
        out['bootstrap_positive_pawn_fraction']=float(np.mean(bs[:,0]>1e-3))
        stable=bool(np.all(bs[:,0]>1e-3) and raw[0]>1e-3)
        out['normalization_stable']=stable
        out['ratio_ci95']=np.quantile(bs/bs[:,0,None],[.025,.975],axis=0).tolist() if stable else None
        if not stable: out['ratios']=None
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    all_ds, all_lab={},{}
    for split in ['train','validation','test']:
        z=np.load(RUN/f'{split}.npz')
        d=Dataset(**{k:z[k] for k in ['x','offsets','targets','games','ctx']})
        rows=json.loads((RUN/f'{split}_positions.json').read_text())
        assert np.array_equal(d.games,[r['game_id'] for r in rows])
        all_ds[split]=d
        all_lab[split]=labels(rows,d.ctx)
    models=json.loads((RUN/'models.json').read_text())
    stage_summ=json.loads((RUN.parent/'stage_analysis'/'summary.json').read_text())
    # Model and regularization comparisons use precisely the same stage contexts.
    families=[]
    for name in ['global','phase','openness','joint']:
        m=models[name]
        full=np.zeros(18);full[m['cols']]=m['weights']
        families.append(dict(model=name,ridge=m['ridge'],estimates=[
            dict(stage=r['stage'],**normalized(material(full,r['mean_context'])))
            for r in stage_summ['estimates']]))
    ridge=[]
    for grid in models['joint']['grid']:
        ridge.append(dict(ridge=grid['ridge'],validation_nll=grid['validation']['game_nll'],
            estimates=[dict(stage=r['stage'],**normalized(material(np.array(grid['weights']),r['mean_context'])))
                       for r in stage_summ['estimates']]))
    independent=[]
    for k,name in enumerate(STAGES):
        tr,va,te=[subset(all_ds[s],all_lab[s]==k) for s in ['train','validation','test']]
        best=None;grid=[]
        for lam in [1e-4,1e-3,1e-2]:
            w,status=optimize(tr,list(range(8)),lam)
            val,_=va.evaluate(w,list(range(8)))
            grid.append(dict(ridge=lam,validation=val,weights=w.tolist(),optimization=status))
            if best is None or val['game_nll']<best[0]:best=(val['game_nll'],lam,w)
        _,lam,w=best
        rng=np.random.default_rng(98710+k)
        boots=[]
        for i in range(100):
            count=np.bincount(rng.integers(tr.ngames,size=tr.ngames),minlength=tr.ngames)
            weights=tr.weights*count[tr.game_index]
            b,_=optimize(tr,list(range(8)),lam,weights,w)
            boots.append(b)
            if (i+1)%25==0:print(f'{name}: bootstrap {i+1}/100',flush=True)
        np.save(OUT/f'{name}_bootstrap.npy',np.array(boots))
        performance={}
        # Diagnostic comparisons only: no model choice based on these test metrics.
        performance['independent'],_=te.evaluate(w,list(range(8)))
        for model in ['global','joint']:
            m=models[model]
            performance[model],_=te.evaluate(np.array(m['weights']),m['cols'])
        record=dict(stage=name,ridge=lam,weights=w.tolist(),grid=grid,
                    training_games=tr.ngames,training_positions=len(tr.offsets),
                    validation_games=va.ngames,test_games=te.ngames,
                    estimates=normalized(w,boots),test=performance)
        independent.append(record)
        save(OUT/f'{name}.json',record)
        print(name,record['estimates'],flush=True)
    results=dict(status='exploratory_followup_after_original_results',
                 distinct_algorithms=False,algorithm='horizon_one_maximum_entropy_IRL',
                 model_families=families,regularization=ridge,independent_stage_models=independent,
                 bootstrap_per_independent_stage=100,elapsed_seconds=time.monotonic()-start,
                 code_sha256=digest(__file__),original_models_sha256=digest(RUN/'models.json'))
    save(OUT/'summary.json',results)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
