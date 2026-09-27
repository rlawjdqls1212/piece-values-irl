"""Post-hoc game-stage summaries of the frozen contextual chess IRL model.

This does not fit three independent IRL models. Context-reference means are
held fixed when propagating the existing training-game bootstrap coefficients.
"""
from pathlib import Path
import csv
import json
import numpy as np
from study import Dataset, NAMES, save, digest

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs' / 'main_fit'
OUT = ROOT / 'runs' / 'stage_analysis'
STAGES = ['opening', 'middlegame', 'endgame']
LABELS = ['오프닝', '미들게임', '엔드게임']


def labels(rows, ctx, opening_moves=10, endgame_units=6):
    # Endgame takes precedence even if an unusually early liquidation occurred.
    return np.array([2 if c[0] <= endgame_units / 24 + 1e-12 else
                     0 if r['ply'] < 2 * opening_moves else 1
                     for r, c in zip(rows, ctx)])


def game_mean(values, games):
    return np.mean([values[games == g].mean(axis=0) for g in np.unique(games)], axis=0)


def material(w, ctx):
    return w[..., :5] + (ctx[0] - .5) * w[..., 8:13] + (ctx[1] - .5) * w[..., 13:18]


def summarize(d, rows, w, boots, opening_moves=10, endgame_units=6):
    lab = labels(rows, d.ctx, opening_moves, endgame_units)
    result = []
    for k in range(3):
        mask = lab == k
        if not mask.any():
            result.append(dict(stage=STAGES[k], positions=0, games=0))
            continue
        ctx = game_mean(d.ctx[mask], d.games[mask])
        raw = material(w, ctx)
        bs = material(boots, ctx)
        raw_ci = np.quantile(bs, [.025, .975], axis=0)
        stable = bool(raw[0] > 1e-3 and np.all(bs[:, 0] > 1e-3))
        ratios = raw / raw[0] if stable else None
        draws = bs / bs[:, 0, None] if stable else None
        varying = np.zeros(5, dtype=int)
        for i in np.flatnonzero(mask):
            block = d.x[d.offsets[i]:d.offsets[i]+d.lengths[i], :5]
            varying += np.ptp(block, axis=0) > 0
        result.append(dict(stage=STAGES[k], positions=int(mask.sum()),
            games=len(np.unique(d.games[mask])), mean_context=ctx.tolist(),
            raw=raw.tolist(), raw_ci95=raw_ci.tolist(), ratios=ratios.tolist() if stable else None,
            ratio_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if stable else None,
            material_variable_positions=varying.tolist()))
    assert sum(r['positions'] for r in result) == len(rows)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = json.loads((RUN / 'models.json').read_text())
    w = np.array(models['joint']['weights'])
    boots = np.load(RUN / 'coefficient_bootstrap.npy')
    datasets, rows_by_split, counts = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        z = np.load(RUN / f'{split}.npz')
        d = Dataset(**{k: z[k] for k in ['x', 'offsets', 'targets', 'games', 'ctx']})
        rows = json.loads((RUN / f'{split}_positions.json').read_text())
        assert len(rows) == len(d.offsets)
        assert np.array_equal([r['game_id'] for r in rows], d.games)
        lab = labels(rows, d.ctx)
        counts[split] = {STAGES[k]: dict(positions=int((lab == k).sum()),
            games=len(np.unique(d.games[lab == k]))) for k in range(3)}
        datasets[split], rows_by_split[split] = d, rows
    tr, rows = datasets['train'], rows_by_split['train']
    main_result = summarize(tr, rows, w, boots)
    sensitivity = []
    for om, eu in [(8, 6), (12, 6), (10, 4), (10, 8)]:
        sensitivity.append(dict(opening_moves=om, endgame_units=eu,
                               estimates=summarize(tr, rows, w, boots, om, eu)))
    contrasts = []
    # Paired contrasts use the same bootstrap coefficient draw across stages.
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        ca, cb = main_result[a]['mean_context'], main_result[b]['mean_context']
        ba, bb = material(boots, ca), material(boots, cb)
        if np.any(ba[:, 0] <= 1e-3) or np.any(bb[:, 0] <= 1e-3):
            continue
        diff = bb / bb[:, 0, None] - ba / ba[:, 0, None]
        contrasts.append(dict(contrast=f'{STAGES[b]} minus {STAGES[a]}',
            difference=(np.array(main_result[b]['ratios']) - main_result[a]['ratios']).tolist(),
            ci95=np.quantile(diff, [.025, .975], axis=0).tolist()))
    result = dict(method='posthoc_stage_summary_of_frozen_joint_model',
        boundaries=dict(opening_moves=10, endgame_phase_units=6, endgame_precedence=True),
        reference='training states; equal game weight within each stage; ratio of mean raw coefficients',
        uncertainty='100 existing training-game bootstrap coefficient draws; reference contexts held fixed; pointwise exploratory intervals',
        counts=counts, estimates=main_result, paired_contrasts=contrasts, sensitivity=sensitivity,
        source_hashes={name:digest(RUN/name) for name in ['models.json','coefficient_bootstrap.npy','train_positions.json']})
    save(OUT / 'summary.json', result)
    with (OUT / 'values.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['stage','piece','pawn_normalized_weight','lower95','upper95','raw_coefficient','training_positions','training_games','informative_positions'])
        for r in main_result:
            for j in range(5):
                writer.writerow([r['stage'],NAMES[j],r['ratios'][j],r['ratio_ci95'][0][j],
                    r['ratio_ci95'][1][j],r['raw'][j],r['positions'],r['games'],r['material_variable_positions'][j]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for k, r in enumerate(main_result):
        x = np.arange(4) + (k-1)*.22
        y = np.array(r['ratios'][1:])
        lo, hi = np.array(r['ratio_ci95'])[:,1:]
        ax.vlines(x,lo,hi,colors=['#276b8c','#c66b39','#638343'][k],lw=2)
        ax.scatter(x,y,label=STAGES[k].title(),color=['#276b8c','#c66b39','#638343'][k])
    ax.set_xticks(range(4),[n.title() for n in NAMES[1:5]])
    ax.set_ylabel('Action reward coefficient / pawn coefficient')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'stage_values.png',dpi=180)
    plt.close(fig)
    md = ['# 오프닝·미들게임·엔드게임별 IRL 추정 가중치', '',
          '기존 상황별 모델을 동결하고 수행한 사후 분석이다. 구간별로 새 모델을 각각 학습한 결과는 아니다.', '',
          '## 구간 정의', '',
          '- 엔드게임: 양쪽 기물의 N+B+2R+4Q가 6 이하(초기 24의 25% 이하). 우선 적용한다.',
          '- 오프닝: 엔드게임이 아니면서 10수까지(ply < 20). 최초 랜덤 4수는 원래부터 학습에서 제외되어 실제 표본은 5~10수다.',
          '- 미들게임: 나머지 상태.',
          '- 보편적인 체스 국면 경계가 아닌 이번 분석의 조작적 정의다.', '',
          '## 폰을 1로 정규화한 추정치', '',
          '| 기물 | 오프닝 | 미들게임 | 엔드게임 |', '|---|---:|---:|---:|']
    for j, name in enumerate(['폰','나이트','비숍','룩','퀸']):
        md.append('| '+name+' | '+' | '.join(f"{r['ratios'][j]:.2f} [{r['ratio_ci95'][0][j]:.2f}, {r['ratio_ci95'][1][j]:.2f}]" for r in main_result)+' |')
    md += ['', '대괄호는 기존 100회 게임 단위 bootstrap 계수에서 계산한 점별 95% 구간이다. 구간별 기준 포지션 분포는 고정했다. 구간 내에서는 게임별 평균 계수를 먼저 계산하고 게임 간 동일 가중 평균을 구한 뒤 폰 계수로 나눴다. 따라서 포지션별 비율을 단순 평균한 값과는 다르다.', '',
           '| 구간 | 학습 포지션 | 기여한 학습 게임 | 평균 phase | 평균 openness |', '|---|---:|---:|---:|---:|']
    for label,r in zip(LABELS,main_result):
        md.append(f"| {label} | {r['positions']:,} | {r['games']} | {r['mean_context'][0]:.3f} | {r['mean_context'][1]:.3f} |")
    md += ['', '게임 하나가 여러 구간에 기여할 수 있으므로 구간별 게임 수를 합쳐 전체 게임 수로 해석하지 않는다.', '',
           '## 민감도', '', '오프닝 끝을 8수/12수로, 엔드게임 기물 점수 기준을 4/8로 바꾼 추가 분석을 summary.json에 저장했다. 모델을 재학습하거나 경계를 성능에 맞춰 선택하지 않았다.', '',
           '## 해석의 범위', '',
           '현재 한 단계 모델은 장기적인 기물 보존·교환·희생을 설명하지 못한다. 추정치가 1보다 작은 기물이 있어도 폰보다 실제로 약하다는 의미가 아니다. 폰의 계수 자체도 상황에 따라 바뀌므로 다른 기물의 비율 변화는 분자와 분모의 변화가 함께 반영된 것이다.', '',
           '국면과 파일 개방성은 함께 변하며 이 분석은 이들을 통제한 인과 효과가 아니다. 오프닝 데이터는 랜덤 초반 4수 이후의 엔진 대국이므로 사람의 정석 오프닝 전체로 일반화할 수 없다. 체크·메이트·중앙화는 기물 가격과 별도 항이며 여기서는 다섯 기물의 계수만 비교한다.', '',
           '산출물: values.csv, summary.json, stage_values.png. 기존 논문·원 실험 결과는 수정하지 않았다.']
    (OUT/'REPORT_KO.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(estimates=main_result,paired_contrasts=contrasts),indent=2))


if __name__ == '__main__':
    main()
