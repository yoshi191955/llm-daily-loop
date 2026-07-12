"""
analyze_predlog.py — 予測ログの厳密分析（horizon別・クラスタ頑健・ベータ分離・クロスセクション）

使い方:
    python analyze_predlog.py predictions_log.csv

4つの階層で評価する（下に行くほど厳しい＝信頼できる）:
  1. トレード単位     … 見かけの成績。同日内の相関を無視するので過大評価になりがち
  2. セッション単位   … 同日の相関を補正。★これが「本当の」サンプル数
  3. ベータ分離       … AlwaysUp(常に買い)と比較。「上げ相場に乗っただけ」を除外
  4. クロスセクション … 同じ日の lean=up 群 vs lean=down 群の差。★市場ベータが設計上ゼロ

horizon列（1d / 3d）があれば自動で分けて比較する。
"""
from __future__ import annotations
import csv, sys, math, statistics as st
from collections import defaultdict

COST, TAX = 0.5, 0.25


def num(x):
    try:
        return float(str(x).replace('+', '').strip())
    except (ValueError, AttributeError):
        return None


def cost_net(r):
    a = r - COST
    return a * (1 - TAX) if a > 0 else a


def tstat(v):
    if len(v) < 2:
        return 0.0
    sd = st.stdev(v)
    return st.mean(v) / (sd / math.sqrt(len(v))) if sd else 0.0


def analyze(rows, label):
    tr = [r for r in rows if r.get('direction') in ('up', 'down')
          and num(r.get('net_return_pct')) is not None]
    if not tr:
        print(f"\n【{label}】 採点済みトレードなし（3d はまだ満期前かもしれません）")
        return

    net = [num(r['net_return_pct']) for r in tr]
    ret = [num(r['return_pct']) for r in tr]
    hit = [str(r.get('hit', '')).upper() == 'TRUE' for r in tr]

    print("\n" + "=" * 70)
    print(f"【{label}】")
    print("=" * 70)

    # 1) トレード単位
    print(f"■ 1. トレード単位 (n={len(tr)})   ※過大評価になりがち")
    print(f"   方向的中率      : {sum(hit)/len(tr):.1%}    [機械的ベースライン 50.4%]")
    print(f"   コスト後プラス率: {sum(1 for x in net if x>0)/len(net):.1%}")
    print(f"   純益(コスト後)  : 平均 {st.mean(net):+.3f}%   t={tstat(net):+.2f}")

    # 2) セッション単位（クラスタ頑健）
    bys = defaultdict(list)
    for r in tr:
        bys[r['session_date']].append(num(r['net_return_pct']))
    sm = [st.mean(v) for _, v in sorted(bys.items())]
    print(f"\n■ 2. ★セッション単位 (n={len(sm)}日)   ※同日の相関を補正した“本当の”標本")
    print(f"   セッション平均の平均: {st.mean(sm):+.3f}%   t={tstat(sm):+.2f}   (|t|>2 が目安)")
    print(f"   プラスのセッション  : {sum(1 for x in sm if x>0)}/{len(sm)}")
    if len(sm) < 30:
        print(f"   ⚠ 独立セッションが {len(sm)} 日しかない。判定には30日以上必要。")

    # 3) ベータ分離
    raw = [(num(r['return_pct']) if r['direction'] == 'up' else -num(r['return_pct']))
           for r in tr]
    au = [cost_net(x) for x in raw]
    edge = st.mean(net) - st.mean(au)
    print(f"\n■ 3. ベータ分離（“上げ相場に乗っただけ”を除外）")
    print(f"   AlwaysUp(常に買い) : {st.mean(au):+.3f}%")
    print(f"   このシステム       : {st.mean(net):+.3f}%")
    print(f"   → 付加価値         : {edge:+.3f}%/トレード  "
          f"{'○ シグナルに価値あり' if edge > 0 else '× 価値なし'}")
    dn = [r for r in tr if r['direction'] == 'down']
    if dn:
        dh = [str(r.get('hit', '')).upper() == 'TRUE' for r in dn]
        dnet = [num(r['net_return_pct']) for r in dn]
        print(f"   down予測: {sum(dh)}/{len(dn)} 的中 ({sum(dh)/len(dn):.0%})  "
              f"net {st.mean(dnet):+.3f}%   ← AlwaysUpでは原理的に取れない部分")

    # 4) クロスセクション（市場中立）— lean 列が必要
    leaned = [r for r in rows if r.get('lean') in ('up', 'down')
              and num(r.get('return_pct')) is not None]
    if leaned:
        print(f"\n■ 4. ★クロスセクション検定（市場ベータを設計上ゼロにする）")
        spreads = []
        for day, grp in sorted(_group(leaned, 'session_date').items()):
            ups = [num(r['return_pct']) for r in grp if r['lean'] == 'up']
            dns = [num(r['return_pct']) for r in grp if r['lean'] == 'down']
            if ups and dns:
                spreads.append(st.mean(ups) - st.mean(dns))
        if spreads:
            print(f"   long-short スプレッド（up群 − down群）: 平均 {st.mean(spreads):+.3f}%"
                  f"   t={tstat(spreads):+.2f}  (n={len(spreads)}日)")
            print(f"   プラスの日: {sum(1 for x in spreads if x>0)}/{len(spreads)}")
            print(f"   → 同じ日の中の差なので、相場の上下は相殺されている。"
                  f"{'○ 銘柄選択に実力あり' if st.mean(spreads) > 0 else '× 実力を確認できず'}")
        else:
            print("   ⚠ up/down 両方の lean がある日が無く、検定不能")
    else:
        print(f"\n■ 4. クロスセクション検定: `lean` 列が未実装のためスキップ")
        print("      → LOOP_SPEC_v2 の通り全銘柄に lean を付ければ、"
              "市場ベータを構造的に排除した最強の検定が可能になります。")


def _group(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "predictions_log.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"読込: {path}  ({len(rows)} 行)")

    horizons = sorted({r.get('horizon', '') for r in rows if r.get('horizon')})
    if len(horizons) > 1:
        for h in horizons:
            analyze([r for r in rows if r.get('horizon') == h], f"horizon = {h}")
        print("\n" + "=" * 70)
        print("■ 1日 vs 3日 の比較 → セッション単位のt値と付加価値が高い方を採用")
        print("=" * 70)
    else:
        analyze(rows, f"horizon = {horizons[0] if horizons else '(単一)'}")


if __name__ == "__main__":
    main()
