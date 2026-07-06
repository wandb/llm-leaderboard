# SWE-Bench Pro repo規模と実行コストの関係

作成日: 2026-06-30

## 前提

- Public件数: 731問
- 現行leaderboard subset: 80問
- repo size metadata: 11 repo
- 観測OpenClaw実行ログ: 134 run、10 repo
- 追加の有料API呼び出し: なし
- 再実行コマンド: `python3 scripts/analysis/analyze_swebench_pro_repo_size_vs_cost.py`
- 出力CSV: `temp/swebench_pro_repo_size_vs_cost_analysis.csv`
- 出力JSON: `temp/swebench_pro_repo_size_vs_cost_analysis.json`

観測ログに `tutao/tutanota` は出ていません。そのため、Public件数のサンプリング判断には含め、観測コスト寄与は0として扱っています。

## 結論

repo規模と観測コストの相関は弱いです。むしろ現在の134 runでは、repo sizeが大きいほどinput tokenが増えるという関係は確認できません。

run単位の相関:

| x | y | n | Pearson | Spearman |
|---|---|---:|---:|---:|
| repo text tokens | observed input | 134 | -0.125 | -0.083 |
| repo source tokens | observed input | 134 | -0.122 | -0.046 |
| tracked MB | observed input | 134 | -0.117 | -0.213 |
| tracked files | observed input | 134 | -0.103 | -0.077 |
| repo text tokens | observed tool calls | 134 | -0.103 | -0.085 |

repo平均単位でも、text token規模とmean inputのSpearmanは -0.236 です。サンプルが10 repoなので過度な一般化はできませんが、「巨大repoだから必ず高コスト」とは言えません。

## repo別の実績

| repo | Public n | 観測n | text tokens M | tracked MB | mean input | p90 input | max input | mean tools | max tools |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protonmail/webclients | 65 | 7 | 31.574 | 128.3 | 377,396 | 1,050,945 | 1,855,228 | 40.3 | 102 |
| gravitational/teleport | 76 | 12 | 22.346 | 268.3 | 407,662 | 897,114 | 1,706,423 | 36.6 | 137 |
| element-hq/element-web | 56 | 13 | 10.966 | 49.6 | 471,476 | 989,344 | 2,093,945 | 49.2 | 116 |
| NodeBB/NodeBB | 44 | 5 | 4.441 | 16.1 | 306,364 | 498,248 | 511,466 | 40.4 | 87 |
| ansible/ansible | 96 | 18 | 3.330 | 14.3 | 1,280,412 | 1,864,149 | 12,417,148 | 50.4 | 283 |
| internetarchive/openlibrary | 91 | 9 | 2.788 | 24.8 | 304,044 | 972,033 | 1,466,620 | 54.1 | 90 |
| flipt-io/flipt | 85 | 25 | 2.200 | 12.5 | 2,198,152 | 6,605,227 | 12,669,591 | 64.3 | 186 |
| qutebrowser/qutebrowser | 79 | 22 | 1.884 | 16.2 | 261,564 | 576,392 | 1,100,472 | 34.3 | 186 |
| navidrome/navidrome | 57 | 10 | 1.173 | 9.8 | 938,349 | 2,491,007 | 6,031,975 | 52.1 | 111 |
| future-architect/vuls | 62 | 13 | 0.732 | 3.0 | 1,020,551 | 2,285,980 | 3,937,215 | 53.2 | 118 |

最大級のバーストは、巨大repoではなく `flipt-io/flipt`、`ansible/ansible`、`navidrome/navidrome`、`future-architect/vuls` に集中しています。`flipt` はrepo text tokensが2.2Mしかありませんが、mean inputは2.20M、max inputは12.67Mです。

## 巨大repo除外の効果

| scenario | 除外repo | Public除外 | Public除外率 | 観測input残存率 | 観測cost残存率 |
|---|---|---:|---:|---:|---:|
| top1 text repo除外 | protonmail/webclients | 65 | 8.9% | 97.9% | 98.2% |
| top3 text repo除外 | protonmail/webclients; gravitational/teleport; element-hq/element-web | 197 | 26.9% | 89.0% | 87.5% |
| top4 text repo除外 | protonmail/webclients; gravitational/teleport; element-hq/element-web; tutao/tutanota | 217 | 29.7% | 89.0% | 87.5% |
| text tokens >10M除外 | protonmail/webclients; gravitational/teleport; element-hq/element-web | 197 | 26.9% | 89.0% | 87.5% |
| text tokens >2M除外 | top8 repo | 533 | 72.9% | 22.8% | 28.3% |

top3/top4の巨大repo除外は、Public問題数を約27-30%減らしますが、観測inputは約11%、観測costは約12.5%しか減りません。つまり、巨大repo除外だけではコスト爆発の主因を抑えられません。

一方で、text tokens >2Mを全部除外すると観測input/costは大きく減りますが、Public 731問中533問、72.9%を捨てることになります。残るrepoもかなり狭くなり、リーダーボードとしての代表性が落ちます。

## 判断

現時点の方針は以下が妥当です。

1. サンプリングは必要。Public 731問をそのまま本番評価するのは費用的にも運用的にも重い。
2. 巨大repo除外は、コスト削減の主策ではなく、静的リスクと運用負荷を下げるための補助策として使う。
3. コスト爆発の主策は、前回の実績ベースシミュレーション通り `1M input cap + 60 tool-call cap` のようなruntime capにする。
4. subsetは `Compact80 + 1M input cap + 60 tool-call cap` を第一候補にする。Compact40は最終手段。
5. 巨大repoは完全除外ではなく、per-repo capと静的cost proxyで露出を制限する方がよい。例: top3/top4 repoは各repo少数に制限し、全体のカテゴリ/言語/難度分布を維持する。

このデータからは、「巨大repoだから高コスト」というより、「小規模repoでもagentが詰まると爆発する」という見方が正しいです。したがって、サンプリング設計とruntime capを併用する必要があります。
