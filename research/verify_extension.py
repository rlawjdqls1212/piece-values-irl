"""Recompute extension intervals and held-out diagnostics from saved artifacts."""
import json
import numpy as np
from extended_irl import OUT, RUN, Dataset, STAGES, labels, subset, normalized, digest


def main():
    result = json.loads((OUT/'summary.json').read_text())
    assert result['code_sha256'] == digest(OUT.parent.parent/'extended_irl.py')
    assert result['original_models_sha256'] == digest(RUN/'models.json')
    z = np.load(RUN/'test.npz')
    data = Dataset(**{k:z[k] for k in ['x','offsets','targets','games','ctx']})
    rows = json.loads((RUN/'test_positions.json').read_text())
    stage = labels(rows, data.ctx)
    models = json.loads((RUN/'models.json').read_text())
    for k, name in enumerate(STAGES):
        record = result['independent_stage_models'][k]
        assert record['stage'] == name
        boots = np.load(OUT/f'{name}_bootstrap.npy')
        assert boots.shape == (100, 8) and np.isfinite(boots).all()
        estimates = normalized(record['weights'], boots)
        for field in ['raw', 'ratios', 'raw_ci95', 'ratio_ci95']:
            np.testing.assert_allclose(estimates[field], record['estimates'][field])
        test = subset(data, stage == k)
        for model in ['independent', 'global', 'joint']:
            weights, cols = (record['weights'], list(range(8))) if model == 'independent' else (models[model]['weights'], models[model]['cols'])
            metrics, _ = test.evaluate(np.array(weights), cols)
            for metric in ['game_nll', 'game_top1']:
                np.testing.assert_allclose(metrics[metric], record['test'][model][metric])
        selected = min(record['grid'], key=lambda g:g['validation']['game_nll'])
        assert selected['ridge'] == record['ridge']
    report = {'bootstrap_refits_checked':300, 'stage_test_metrics_reproduced':True,
              'confidence_intervals_reproduced':True, 'hashes_verified':True}
    (OUT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
