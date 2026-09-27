"""Common-position Stockfish strength-setting labels and contextual IRL."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
import chess
import chess.engine
from study import Dataset, optimize, save, digest, COLS, features
from extended_irl import normalized
from stage_analysis import material

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/main_fit'
OUT=ROOT/'runs/elo_irl'
LEVELS=[1500,2000,2500]
ENGINE=ROOT/'engine/stockfish/stockfish-windows-x86-64.exe'

def collect(level):
    dest=OUT/str(level);dest.mkdir(parents=True,exist_ok=True)
    engine=chess.engine.SimpleEngine.popen_uci(str(ENGINE))
    engine.configure({'Threads':1,'Hash':64,'UCI_LimitStrength':True,'UCI_Elo':level})
    t=time.monotonic()
    try:
        for split in ['train','validation','test']:
            rows=json.loads((RUN/f'{split}_positions.json').read_text())
            path=dest/f'{split}_labels.jsonl'
            saved=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            assert all(r['fen']==rows[i]['fen'] and r['game_id']==rows[i]['game_id'] for i,r in enumerate(saved))
            with path.open('a',encoding='utf-8') as stream:
                for i,r in enumerate(rows[len(saved):],len(saved)):
                    board=chess.Board(r['fen'])
                    move=engine.play(board,chess.engine.Limit(nodes=5000),game=object()).move
                    assert move in board.legal_moves
                    stream.write(json.dumps(dict(game_id=r['game_id'],fen=r['fen'],ply=r['ply'],move=move.uci()))+'\n')
                    if (i+1)%1000==0:stream.flush();print(f'Elo {level} {split}: {i+1}/{len(rows)}',flush=True)
            assert len(path.read_text().splitlines())==len(rows)
    finally:engine.quit()
    save(dest/'collection.json',dict(level=level,engine_sha256=digest(ENGINE),nodes=5000,hash_mb=64,threads=1,
          clear_per_position=True,elapsed_seconds=time.monotonic()-t,
          labels_sha256={s:digest(dest/f'{s}_labels.jsonl') for s in ['train','validation','test']}))
    print('Collected',level,flush=True)

def datasets(level):
    ds={}
    for split in ['train','validation','test']:
        z=np.load(RUN/f'{split}.npz');kw={k:z[k] for k in ['x','offsets','targets','games','ctx']}
        rows=json.loads((RUN/f'{split}_positions.json').read_text())
        labels=[json.loads(s) for s in (OUT/str(level)/f'{split}_labels.jsonl').read_text().splitlines()]
        assert len(rows)==len(labels)
        indices=[]
        for i,(r,l) in enumerate(zip(rows,labels)):
            assert r['fen']==l['fen'] and r['game_id']==l['game_id']
            board=chess.Board(r['fen'])
            # Cached study.candidates rows use UCI lexical order, not generator order.
            moves=sorted(board.legal_moves,key=lambda m:m.uci())
            assert kw['targets'][i]==kw['offsets'][i]+moves.index(chess.Move.from_uci(r['move']))
            target=kw['offsets'][i]+moves.index(chess.Move.from_uci(l['move']))
            color=board.turn;before=features(board,color)
            board.push_uci(l['move'])
            np.testing.assert_allclose(kw['x'][target,:8],features(board,color)-before)
            indices.append(target)
        kw['targets']=np.array(indices);ds[split]=Dataset(**kw)
    return ds

def fit(level):
    dest=OUT/str(level);ds=datasets(level);models={}
    start=time.monotonic()
    for name in ['global','joint']:
        grid=[]
        for lam in [1e-4,1e-3,1e-2]:
            w,st=optimize(ds['train'],COLS[name],lam)
            val,_=ds['validation'].evaluate(w,COLS[name])
            grid.append(dict(ridge=lam,weights=w.tolist(),validation=val,optimization=st))
        best=min(grid,key=lambda g:g['validation']['game_nll'])
        te,_=ds['test'].evaluate(np.array(best['weights']),COLS[name])
        models[name]=dict(**best,grid=grid,test=te,cols=COLS[name])
    tr=ds['train'];m=models['joint'];rng=np.random.default_rng(2026092802);boots=[]
    for i in range(100):
        mult=np.bincount(rng.integers(tr.ngames,size=tr.ngames),minlength=tr.ngames)
        b,_=optimize(tr,COLS['joint'],m['ridge'],tr.weights*mult[tr.game_index],np.array(m['weights']))
        boots.append(b)
        if (i+1)%25==0:print(f'Elo {level}: bootstrap {i+1}/100',flush=True)
    boots=np.array(boots);np.save(dest/'joint_bootstrap.npy',boots)
    stage=json.loads((ROOT/'runs/stage_analysis/summary.json').read_text())['estimates']
    count=json.loads((ROOT/'runs/piece_count_irl/summary.json').read_text())['groups']
    refs=[dict(name=r['stage'],context=r['mean_context'],kind='stage') for r in stage]
    refs += [dict(name=r['group'],context=r['coverage']['train']['context_mean'],kind='piece_count') for r in count]
    refs += [dict(name='all',context=(tr.weights@tr.ctx).tolist(),kind='overall')]
    estimates=[]
    for ref in refs:
        w=material(np.array(m['weights']),ref['context'])
        b=np.array([material(v,ref['context']) for v in boots])
        estimates.append(dict(**ref,**normalized(w,b)))
    result=dict(level=level,models=models,estimates=estimates,bootstrap=100,
                code_sha256=digest(__file__),protocol_sha256=digest(ROOT/'ELO_PROTOCOL.md'),
                elapsed_seconds=time.monotonic()-start)
    save(dest/'summary.json',result)
    print('Fitted',level,flush=True)

def summarize():
    results=[json.loads((OUT/str(n)/'summary.json').read_text()) for n in LEVELS]
    test=[datasets(n)['test'] for n in LEVELS]
    agreement=[]
    for i,a in enumerate(LEVELS):
        for j,b in enumerate(LEVELS):
            if j<=i:continue
            eq=test[i].targets==test[j].targets
            agreement.append(dict(a=a,b=b,game_weighted=float(test[i].weights@eq),positions=float(eq.mean())))
    save(OUT/'summary.json',dict(levels=results,test_label_agreement=agreement,
          interpretation='Stockfish UCI_Elo settings, not measured human Elo',paired_common_positions=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','fit','summarize']);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='summarize':summarize()
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            list(pool.map(collect if a.command=='collect' else fit,LEVELS))
