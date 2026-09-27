"""Reproducible contextual-bandit maximum-entropy chess reward study.

No engine evaluation scores are used as labels. See PROTOCOL.md for scope.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar
import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent
NAMES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'center', 'check', 'mate']
HAND = np.array([1., 3., 3.2, 5., 9., .6, .5, 15.])
COLS = {'global': list(range(8)), 'phase': list(range(13)),
        'openness': list(range(8)) + list(range(13, 18)), 'joint': list(range(18))}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(board):
    phase = sum(len(board.pieces(k, c)) * v for c in chess.COLORS
                for k, v in [(2, 1), (3, 1), (4, 2), (5, 4)]) / 24
    closed_files = sum(bool(board.pawns & board.occupied_co[0] & f) and
                       bool(board.pawns & board.occupied_co[1] & f) for f in chess.BB_FILES)
    return np.array([min(1., phase), 1 - closed_files / 8])


def features(board, color):
    counts = [len(board.pieces(k, color)) - len(board.pieces(k, not color)) for k in range(1, 6)]
    center = sum((1 if p.color == color else -1) *
                 (7 - abs(chess.square_file(s) - 3.5) - abs(chess.square_rank(s) - 3.5)) / 7
                 for s, p in board.piece_map().items())
    # Only a checked opponent is a delivered check, including the pre-move baseline.
    check = board.turn != color and board.is_check()
    mate = check and board.is_checkmate()
    return np.array(counts + [center, float(check), float(mate)])


def candidates(board):
    color = board.turn
    ctx = context(board)
    before = features(board, color)
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    x = []
    for move in moves:
        board.push(move)
        delta = features(board, color) - before
        board.pop()
        x.append(np.r_[delta, delta[:5] * (ctx[0] - .5), delta[:5] * (ctx[1] - .5)])
    return moves, np.asarray(x), ctx


def collect(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine_path = Path(args.engine).resolve()
    config = dict(games=args.games, seed=args.seed, nodes=args.nodes, plies=args.plies,
                  opening_plies=8, stride=2, threads=1, hash_mb=64,
                  engine_sha256=digest(engine_path), engine_path=str(engine_path))
    cp = out / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Existing collection has different config; use a new output directory')
    save(cp, config)
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=120)
    engine.configure({'Threads': 1, 'Hash': 64})
    save(out / 'environment.json', dict(engine=engine.id, python=platform.python_version(),
         numpy=np.__version__, scipy=scipy.__version__, chess=chess.__version__, platform=platform.platform()))
    start = time.monotonic()
    try:
        for i in range(args.games):
            path = out / 'games' / f'{i:05d}.json'
            if path.exists():
                continue
            t0 = time.monotonic()
            rng = np.random.default_rng(args.seed + i)
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update(Event='Contextual piece-value study', White=engine.id['name'],
                                Black=engine.id['name'], Round=str(i), Seed=str(args.seed+i))
            node = game
            engine.configure({'Clear Hash': None})
            token = object()
            rows, history = [], []
            for ply in range(args.plies):
                if board.is_game_over(claim_draw=True):
                    break
                legal = sorted(board.legal_moves, key=lambda m: m.uci())
                if ply < 8:
                    move = legal[int(rng.integers(len(legal)))]
                else:
                    move = engine.play(board, chess.engine.Limit(nodes=args.nodes), game=token).move
                    if move not in legal:
                        raise AssertionError('Illegal engine move')
                    if (ply - 8) % 2 == i % 2:
                        rows.append(dict(fen=board.fen(), move=move.uci(), ply=ply,
                                         legal_count=len(legal), context=context(board).tolist()))
                history.append(move.uci())
                board.push(move)
                node = node.add_variation(move)
            outcome = board.outcome(claim_draw=True)
            result = outcome.result() if outcome else '*'
            reason = outcome.termination.name if outcome else 'TRUNCATED'
            game.headers['Result'] = result
            game.headers['Termination'] = reason
            record = dict(game_id=i, seed=args.seed+i, moves=history, positions=rows,
                          result=result, termination=reason, elapsed_seconds=time.monotonic()-t0,
                          pgn=str(game))
            save(path, record)
            print(f'game {i+1}/{args.games}, {len(history)} plies, {len(rows)} positions, '
                  f'{reason}, {record["elapsed_seconds"]:.1f}s', flush=True)
    finally:
        engine.quit()
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'games').glob('*.json'))]
    (out/'games.pgn').write_text('\n\n'.join(r['pgn'] for r in records)+'\n', encoding='utf-8')
    save(out/'collection_summary.json', dict(games=len(records),
         positions=sum(len(r['positions']) for r in records), wall_seconds_this_call=time.monotonic()-start,
         total_game_seconds=sum(r['elapsed_seconds'] for r in records),
         game_files={p.name: digest(p) for p in sorted((out/'games').glob('*.json'))}))


class Dataset:
    def __init__(self, x, offsets, targets, games, ctx):
        self.x = np.asarray(x, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=int)
        self.targets = np.asarray(targets, dtype=int)
        self.games = np.asarray(games, dtype=int)
        self.ctx = np.asarray(ctx, dtype=float)
        self.lengths = np.diff(np.r_[self.offsets, len(self.x)])
        self.owner = np.repeat(np.arange(len(self.offsets)), self.lengths)
        _, self.game_index = np.unique(self.games, return_inverse=True)
        self.ngames = len(np.unique(self.games))
        counts = np.bincount(self.game_index)
        self.weights = 1 / counts[self.game_index] / self.ngames

    def logits(self, w, cols):
        return self.x[:, cols] @ w

    def objective(self, w, cols, ridge, weights=None):
        weights = self.weights if weights is None else weights
        x = self.x[:, cols]
        z = x @ w
        peak = np.maximum.reduceat(z, self.offsets)
        exp = np.exp(z - peak[self.owner])
        den = np.add.reduceat(exp, self.offsets)
        probs = exp / den[self.owner]
        loss = peak + np.log(den) - z[self.targets]
        residual = probs * weights[self.owner]
        residual[self.targets] -= weights
        return float(weights @ loss + ridge * .5 * (w @ w)), x.T @ residual + ridge*w

    def evaluate(self, w, cols):
        z = self.logits(w, cols)
        peak = np.maximum.reduceat(z, self.offsets)
        den = np.add.reduceat(np.exp(z-peak[self.owner]), self.offsets)
        nll = peak+np.log(den)-z[self.targets]
        ties = np.isclose(z, peak[self.owner], atol=1e-9, rtol=1e-7)
        top = ties[self.targets] / np.add.reduceat(ties.astype(float), self.offsets)
        gnll = np.bincount(self.game_index, weights=nll) / np.bincount(self.game_index)
        return dict(game_nll=float(self.weights@nll), position_nll=float(nll.mean()),
                    game_top1=float(self.weights@top), positions=len(nll), games=self.ngames), gnll


def prepare(data_dir, out):
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((Path(data_dir)/'games').glob('*.json'))]
    if len(records) < 10:
        raise ValueError('Need at least ten completed games')
    order = np.random.default_rng(20260927).permutation(len(records))
    n = len(order)
    groups = dict(train=order[:int(.6*n)], validation=order[int(.6*n):int(.8*n)], test=order[int(.8*n):])
    seen, datasets, audit = set(), {}, {}
    for split, indices in groups.items():
        blocks, offsets, targets, games, ctx, retained = [], [], [], [], [], []
        total = 0
        dropped = dict(forced=0, duplicate=0)
        for idx in indices:
            r = records[idx]
            # Replay all moves once to verify stored states and labels against history.
            replay = chess.Board()
            positions_by_ply = {p['ply']: p for p in r['positions']}
            for ply, uci in enumerate(r['moves']):
                if ply in positions_by_ply:
                    assert replay.fen() == positions_by_ply[ply]['fen']
                    assert uci == positions_by_ply[ply]['move']
                replay.push_uci(uci)
            for p in r['positions']:
                board = chess.Board(p['fen'])
                key = ' '.join(board.fen().split()[:4])
                if p['legal_count'] <= 1:
                    dropped['forced'] += 1
                    continue
                if key in seen:
                    dropped['duplicate'] += 1
                    continue
                seen.add(key)
                moves, x, c = candidates(board)
                assert len(moves) == p['legal_count']
                target = moves.index(chess.Move.from_uci(p['move']))
                blocks.append(x)
                offsets.append(total)
                targets.append(total+target)
                total += len(x)
                games.append(r['game_id'])
                ctx.append(c)
                retained.append(dict(game_id=r['game_id'], ply=p['ply'], fen=p['fen'], move=p['move']))
        if not blocks:
            raise ValueError(f'Empty split: {split}')
        d = Dataset(np.vstack(blocks), offsets, targets, games, ctx)
        datasets[split] = d
        np.savez_compressed(out/f'{split}.npz', x=d.x, offsets=d.offsets, targets=d.targets,
                            games=d.games, ctx=d.ctx)
        save(out/f'{split}_positions.json', retained)
        audit[split] = dict(assigned_games=[records[i]['game_id'] for i in indices],
                           retained_games=d.ngames, positions=len(games), candidates=total, dropped=dropped,
                           material_variable_positions=int(sum(np.any(np.ptp(b[:,:5], axis=0)>0) for b in blocks)),
                           delivered_mate_candidates=int(np.sum(d.x[:,7]>0)))
        print(f'{split}: {d.ngames} games, {len(games)} positions, {total} candidates', flush=True)
    save(out/'split_audit.json', audit)
    return datasets, audit


def optimize(d, cols, ridge, weights=None, initial=None):
    w = np.zeros(len(cols)) if initial is None else initial
    result = minimize(d.objective, w, args=(cols, ridge, weights), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
    if not result.success:
        raise RuntimeError(str(result.message))
    return result.x, dict(iterations=result.nit, objective=float(result.fun), message=str(result.message))


def fit(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    ds, audit = prepare(args.data, out)
    tr, va, te = [ds[k] for k in ['train', 'validation', 'test']]
    models, report, losses = {}, {}, {}
    for name, cols in COLS.items():
        grid = []
        best = None
        for ridge in [1e-4, 1e-3, 1e-2]:
            w, status = optimize(tr, cols, ridge)
            val, _ = va.evaluate(w, cols)
            grid.append(dict(ridge=ridge, validation=val, weights=w.tolist(), **status))
            if best is None or val['game_nll'] < best[0]:
                best = val['game_nll'], w, ridge
        _, w, ridge = best
        models[name] = dict(cols=cols, weights=w.tolist(), ridge=ridge, grid=grid)
        print(f'{name}: validation NLL={best[0]:.5f}, ridge={ridge}', flush=True)
    # This temperature is fitted on training data, not selected using test scores.
    temp = minimize_scalar(lambda t: tr.evaluate(HAND*np.exp(t), list(range(8)))[0]['game_nll'],
                           bounds=(-6, 6), method='bounded')
    models['handcrafted'] = dict(cols=list(range(8)), weights=(HAND*np.exp(temp.x)).tolist(),
                                 inverse_temperature=float(np.exp(temp.x)))
    models['uniform'] = dict(cols=list(range(8)), weights=np.zeros(8).tolist())
    save(out/'models.json', models)
    for name, m in models.items():
        report[name], losses[name] = te.evaluate(np.array(m['weights']), m['cols'])
        print(name, report[name], flush=True)
    rng = np.random.default_rng(440)
    differences = losses['joint']-losses['global']
    bootdiff = [float(rng.choice(differences, len(differences), replace=True).mean()) for _ in range(2000)]
    primary = dict(contrast='joint minus global; negative favors joint', mean=float(differences.mean()),
                   ci95=np.quantile(bootdiff,[.025,.975]).tolist(), bootstrap_games=len(differences))
    # Cluster bootstrap of training games, keeping the selected ridge fixed.
    boots = []
    m = models['joint']
    for b in range(args.bootstrap):
        counts = np.bincount(rng.integers(tr.ngames, size=tr.ngames), minlength=tr.ngames)
        weights = tr.weights*counts[tr.game_index]
        w, _ = optimize(tr, m['cols'], m['ridge'], weights, np.array(m['weights']))
        boots.append(w)
        if (b+1) % 10 == 0:
            print(f'coefficient bootstrap {b+1}/{args.bootstrap}', flush=True)
    if not boots:
        raise ValueError('At least one coefficient bootstrap required')
    np.save(out/'coefficient_bootstrap.npy', np.asarray(boots))
    cells = []
    phase_edges = np.quantile(tr.ctx[:,0], [1/3,2/3])
    open_edges = np.quantile(tr.ctx[:,1], [1/3,2/3])
    labels_p = np.searchsorted(phase_edges, tr.ctx[:,0], side='right')
    labels_o = np.searchsorted(open_edges, tr.ctx[:,1], side='right')
    for pi in range(3):
        for oi in range(3):
            ix = (labels_p==pi)&(labels_o==oi)
            ng = len(np.unique(tr.games[ix]))
            if ng < min(20, tr.ngames):
                continue
            c = tr.ctx[ix].mean(axis=0)
            def material(w):
                w=np.asarray(w)
                return w[..., :5]+(c[0]-.5)*w[...,8:13]+(c[1]-.5)*w[...,13:18]
            raw=material(m['weights'])
            bs=material(boots)
            ci=np.quantile(bs,[.025,.975],axis=0)
            stable=bool(ci[0,0]>1e-3 and raw[0]>1e-3)
            cells.append(dict(phase_bin=pi, openness_bin=oi, context=c.tolist(), games=ng,
                 positions=int(ix.sum()), raw=raw.tolist(), raw_ci95=ci.tolist(), pawn_ratio_stable=stable,
                 ratios=(raw/raw[0]).tolist() if stable else None,
                 ratio_ci95=np.quantile(bs[bs[:,0]>1e-3]/bs[bs[:,0]>1e-3,0,None],[.025,.975],axis=0).tolist() if stable else None))
    summary=dict(test=report, primary=primary, contexts=cells, bootstrap_refits=len(boots),
                 phase_edges=phase_edges.tolist(), openness_edges=open_edges.tolist(),
                 train_context_correlation=float(np.corrcoef(tr.ctx.T)[0,1]),
                 audit=audit, elapsed_seconds=time.monotonic()-start, method='horizon_one_maximum_entropy_IRL',
                 data_config=json.loads((Path(args.data)/'config.json').read_text()),
                 data_summary_sha256=digest(Path(args.data)/'collection_summary.json'),
                 code_sha256=digest(__file__), protocol_sha256=digest(ROOT/'PROTOCOL.md'))
    save(out/'summary.json',summary)
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['model']+list(next(iter(report.values()))))
        writer.writeheader()
        for name,row in report.items(): writer.writerow(dict(model=name,**row))
    plots(out, summary)
    print('Saved', out.resolve(), flush=True)


def plots(out, s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['uniform','handcrafted','global','phase','openness','joint']
    fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
    ax.bar(names,[s['test'][n]['game_nll'] for n in names],color=['#9aa2ad']*2+['#276b8c']*3+['#c66b39'])
    ax.set_ylabel('Held-out game-mean NLL (lower is better)')
    fig.savefig(out/'prediction.pdf'); fig.savefig(out/'prediction.png',dpi=150); plt.close(fig)
    if s['contexts']:
        cells=s['contexts']
        fig,axes=plt.subplots(1,5,figsize=(11,3.6),layout='constrained',sharex=True)
        for k,ax in enumerate(axes):
            y=np.array([c['raw'][k] for c in cells])
            lo=np.array([c['raw_ci95'][0][k] for c in cells])
            hi=np.array([c['raw_ci95'][1][k] for c in cells])
            x=np.arange(len(cells))
            ax.vlines(x,lo,hi,color='#276b8c');ax.scatter(x,y,color='#276b8c',s=14)
            ax.axhline(0,color='gray',lw=.6)
            ax.set_title(NAMES[k]); ax.set_xticks(x, [f'{c["phase_bin"]+1}/{c["openness_bin"]+1}' for c in cells],rotation=90)
        axes[0].set_ylabel('Raw reward coefficient (95% cluster CI)')
        fig.supxlabel('Empirical phase/openness cell (1 = low)')
        fig.savefig(out/'coefficients.pdf');fig.savefig(out/'coefficients.png',dpi=150);plt.close(fig)


def selftest():
    b=chess.Board()
    assert np.allclose(context(b),[1,0])
    before=b.fen()
    moves,x,c=candidates(b)
    assert b.fen()==before and len(moves)==20 and np.all(x[:,:5]==0)
    b=chess.Board('4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1')
    moves,x,_=candidates(b)
    assert x[moves.index(chess.Move.from_uci('e4d5')),4]==1
    b=chess.Board('4k3/8/8/4p3/3Q4/8/8/4K3 b - - 0 1')
    moves,x,_=candidates(b)
    assert x[moves.index(chess.Move.from_uci('e5d4')),4]==1
    b=chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    moves,x,_=candidates(b)
    prom=x[moves.index(chess.Move.from_uci('a7a8q'))]
    assert prom[0]==-1 and prom[4]==1
    rng=np.random.default_rng(911)
    x=rng.normal(size=(6000,18))
    offsets=np.arange(0,len(x),6)
    truth=rng.normal(size=18)*.2
    targets=[]
    for o in offsets:
        p=np.exp(x[o:o+6]@truth);p/=p.sum()
        targets.append(o+rng.choice(6,p=p))
    d=Dataset(x,offsets,targets,np.arange(len(offsets))//10,np.zeros((len(offsets),2)))
    w=rng.normal(size=18)*.1
    loss, grad=d.objective(w,list(range(18)),.001)
    numeric=[]
    for k in range(18):
        e=np.zeros(18);e[k]=1e-5
        numeric.append((d.objective(w+e,list(range(18)),.001)[0]-d.objective(w-e,list(range(18)),.001)[0])/2e-5)
    error=float(np.max(np.abs(grad-numeric)))
    assert error<1e-6
    learned,_=optimize(d,list(range(18)),1e-4)
    rmse=float(np.sqrt(np.mean((learned-truth)**2)))
    assert rmse<.1
    assert np.isclose(d.weights.sum(),1)
    save(ROOT/'tests.json',dict(gradient_max_error=error,synthetic_coefficient_rmse=rmse,
          tests=['initial legal candidates','no mutation','white capture','black capture','promotion',
                 'gradient finite difference','synthetic recovery','game weighting']))
    print('PASS',error,rmse)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect')
    c.add_argument('--engine',required=True);c.add_argument('--out',required=True)
    c.add_argument('--games',type=int,default=300);c.add_argument('--seed',type=int,default=10000)
    c.add_argument('--nodes',type=int,default=5000);c.add_argument('--plies',type=int,default=240)
    f=sub.add_parser('fit');f.add_argument('--data',required=True);f.add_argument('--out',required=True)
    f.add_argument('--bootstrap',type=int,default=100)
    sub.add_parser('selftest')
    args=p.parse_args()
    if args.command=='collect':
        if args.games<1 or args.nodes<1 or args.plies<10: p.error('invalid collection size')
        collect(args)
    elif args.command=='fit':
        if args.bootstrap<1: p.error('bootstrap must be positive')
        fit(args)
    else: selftest()
