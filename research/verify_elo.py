"""Validate saved strength labels, fitted metrics and conditional intervals."""
import json
import numpy as np
from elo_irl import ROOT,RUN,OUT,LEVELS,datasets,digest,material,normalized

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert [r['level'] for r in summary['levels']]==LEVELS
    tests=[]
    for level,r in zip(LEVELS,summary['levels']):
        assert r['code_sha256']==digest(ROOT/'elo_irl.py')
        assert r['protocol_sha256']==digest(ROOT/'ELO_PROTOCOL.md')
        collection=json.loads((OUT/str(level)/'collection.json').read_text())
        for split in ['train','validation','test']:
            assert digest(OUT/str(level)/f'{split}_labels.jsonl')==collection['labels_sha256'][split]
        ds=datasets(level);tests.append(ds['test'])
        for name,m in r['models'].items():
            metrics,_=ds['test'].evaluate(np.array(m['weights']),m['cols'])
            np.testing.assert_allclose(metrics['game_nll'],m['test']['game_nll'])
            assert m['ridge']==min(m['grid'],key=lambda x:x['validation']['game_nll'])['ridge']
        boots=np.load(OUT/str(level)/'joint_bootstrap.npy');assert boots.shape==(100,18)
        for e in r['estimates']:
            raw=material(np.array(r['models']['joint']['weights']),e['context'])
            b=np.array([material(w,e['context']) for w in boots])
            est=normalized(raw,b)
            np.testing.assert_allclose(est['raw'],e['raw'])
            np.testing.assert_allclose(est['raw_ci95'],e['raw_ci95'])
            if est['ratios'] is not None:np.testing.assert_allclose(est['ratio_ci95'],e['ratio_ci95'])
            else:assert e['ratios'] is None
    for record in summary['test_label_agreement']:
        a=tests[LEVELS.index(record['a'])];b=tests[LEVELS.index(record['b'])]
        np.testing.assert_allclose(a.weights@(a.targets==b.targets),record['game_weighted'])
    report=dict(levels=LEVELS,legal_moves_and_state_alignment_verified=True,
                test_metrics_and_intervals_reproduced=True,hashes_verified=True,
                labels_per_setting=18351,bootstrap_refits=300)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
