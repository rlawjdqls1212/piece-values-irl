"""Independent replay of saved piece-count estimates, partitions and diagnostics."""
import json
import numpy as np
from piece_count_irl import ROOT,RUN,OUT,GROUPS,count_pieces,Dataset,subset,normalized,digest

def main():
    s=json.loads((OUT/'summary.json').read_text())
    assert s['code_sha256']==digest(ROOT/'piece_count_irl.py')
    assert s['protocol_sha256']==digest(ROOT/'PIECE_COUNT_PROTOCOL.md')
    assert s['original_models_sha256']==digest(RUN/'models.json')
    originals=json.loads((RUN/'models.json').read_text())
    all_games=[]
    for split in ['train','validation','test']:
        z=np.load(RUN/f'{split}.npz');d=Dataset(**{k:z[k] for k in ['x','offsets','targets','games','ctx']})
        rows=json.loads((RUN/f'{split}_positions.json').read_text());count=count_pieces(rows)
        groups=np.where(count<=10,0,np.where(count>=22,2,1))
        assert len(groups)==sum(r['coverage'][split]['positions'] for r in s['groups'])
        all_games.append(set(d.games))
        for k,r in enumerate(s['groups']):
            sub=subset(d,groups==k)
            assert len(sub.offsets)==r['coverage'][split]['positions']
            if split=='test':
                for name in ['independent','global','joint']:
                    w,cols=(r['weights'],list(range(8))) if name=='independent' else (originals[name]['weights'],originals[name]['cols'])
                    met,_=sub.evaluate(np.array(w),cols)
                    np.testing.assert_allclose(met['game_nll'],r['test'][name]['game_nll'])
    assert all(not all_games[i]&all_games[j] for i in range(3) for j in range(i))
    bs=[]
    for name,r in zip(GROUPS,s['groups']):
        b=np.load(OUT/f'{name}_bootstrap.npy');assert b.shape==(100,8)
        est=normalized(r['weights'],b)
        np.testing.assert_allclose(est['ratio_ci95'],r['estimates']['ratio_ci95'])
        assert r['ridge']==min(r['grid'],key=lambda g:g['validation']['game_nll'])['ridge']
        bs.append(b[:,:5]/b[:,0,None])
    np.testing.assert_allclose(np.quantile(bs[0]-bs[2],[.025,.975],axis=0),s['sparse_minus_dense']['ci95'])
    report=dict(tables_and_intervals_verified=True,group_partition_verified=True,test_metrics_reproduced=True,
                hashes_verified=True,bootstrap_refits=300)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
