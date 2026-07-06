# SWE-Bench Pro 実績ベースの実行上限コストシミュレーション

作成日: 2026-06-30

## 前提

- 入力データ: `temp/swebench_pro_observed_usage_distribution.json`
- 対象run数: 134件
- 追加の有料API呼び出し: なし
- 再実行コマンド: `python3 scripts/analysis/simulate_swebench_pro_runtime_caps.py`
- 詳細CSV: `temp/swebench_pro_runtime_cap_simulation_detailed.csv`
- 補助CSV: `temp/swebench_pro_event_proxy_cap_simulation.csv`

現在のローカル記録には、ターンごとの累積トークン列は残っていません。そのため、ここで厳密に扱えるのは「問題単位の総input/output/cacheRead」と「tool call数」です。以下では `tool_calls` を実行ターン上限の主指標として使います。`assistant_events` と `timeline_events` も補助CSVには出していますが、モデル/ランタイムで意味が揺れるため、正式なコスト制御指標にはまだしません。

コスト計算は保守的です。上限にかかった場合、

```text
scale = min(1, token_cap / input_tokens, tool_call_cap / tool_calls)
```

として `input` と `cacheRead` だけを `scale` 倍し、`output` は据え置いています。実際に途中停止すればoutputも減る可能性がありますが、このレポートでは節約額を過大に見積もらない設定にしています。

## 実績分布

| 観測モデル挙動 | n | input p50 | input p90 | input max | tool p50 | tool p90 | tool max |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-v4-pro-thinking-max | 53 | 49,685 | 78,980 | 4,802,538 | 40 | 79 | 116 |
| claude-sonnet-4_6-openrouter-high | 23 | 488,640 | 1,958,848 | 12,669,591 | 34 | 59 | 186 |
| claude-opus-4_7-openrouter-xhigh | 29 | 467,299 | 2,303,517 | 6,674,164 | 25 | 62 | 111 |
| gemini-3_1-pro-preview-openrouter | 18 | 548,556 | 4,560,372 | 12,417,148 | 93 | 176 | 283 |
| qwen3_6-max-preview-openrouter | 11 | 1,320,360 | 3,047,352 | 6,501,821 | 28 | 66 | 111 |

全134件では、input tokenの上位5%が48.2%、上位10%が63.3%、上位20%が78.9%を占めます。したがって、SWE-Bench Proのコスト問題は平均的に高いというより、少数のハマりrunが支配している可能性が高いです。

## 80問換算コスト

| tier | 上限なし | 1M+40 tool | 1M+60 tool | 1M+80 tool | 2M+60 tool | 500k+60 tool | 500k+40 tool |
|---|---:|---:|---:|---:|---:|---:|---:|
| openai-small-like_on_deepseek_behavior | $24.63 | $16.16 | $19.53 | $21.35 | $20.20 | $19.19 | $15.82 |
| low-deepseek-like | $51.44 | $31.14 | $36.83 | $39.89 | $39.55 | $35.46 | $29.78 |
| mid-glm5.2-like_on_deepseek_behavior | $47.27 | $30.43 | $36.65 | $40.01 | $38.21 | $35.88 | $29.65 |
| high-sonnet-like | $364.24 | $177.12 | $178.58 | $178.58 | $229.05 | $131.06 | $131.06 |
| very-high-opus-gpt5.5-like | $538.60 | $290.09 | $290.70 | $290.70 | $392.82 | $218.33 | $218.33 |
| gemini-pro-like | $407.93 | $101.80 | $131.12 | $152.83 | $152.01 | $115.17 | $89.15 |
| qwen-max-like | $171.76 | $85.93 | $85.93 | $85.93 | $124.73 | $49.26 | $49.26 |

## 節約率

| tier | 1M+40 tool | 1M+60 tool | 1M+80 tool | 2M+60 tool | 500k+60 tool | 500k+40 tool |
|---|---:|---:|---:|---:|---:|---:|
| openai-small-like_on_deepseek_behavior | 34.4% | 20.7% | 13.3% | 18.0% | 22.1% | 35.8% |
| low-deepseek-like | 39.5% | 28.4% | 22.5% | 23.1% | 31.1% | 42.1% |
| mid-glm5.2-like_on_deepseek_behavior | 35.6% | 22.5% | 15.4% | 19.2% | 24.1% | 37.3% |
| high-sonnet-like | 51.4% | 51.0% | 51.0% | 37.1% | 64.0% | 64.0% |
| very-high-opus-gpt5.5-like | 46.1% | 46.0% | 46.0% | 27.1% | 59.5% | 59.5% |
| gemini-pro-like | 75.0% | 67.9% | 62.5% | 62.7% | 71.8% | 78.1% |
| qwen-max-like | 50.0% | 50.0% | 50.0% | 27.4% | 71.3% | 71.3% |

## 解釈

1M input capは、Sonnet/Opus級では費用削減の主因です。Sonnetでは23件中7件、Opusでは29件中7件だけを制限して、80問換算コストをそれぞれ約51%/46%下げます。これは、問題数を80から40へ機械的に落とす前に試す価値があります。

tool call cap単独は、モデルによって効果が違います。Sonnet/Opus級では40 tool cap単独でも削減効果はありますが、1M input capを入れると40/60/80 toolの差は小さくなります。一方、DeepSeek挙動では40 tool capが53件中26件に当たり、inputが小さいrunまで止めます。安いモデルの探索能力を落とす割に節約額は限定的な可能性があります。

500k input capはかなり強いです。高額tierのコストは大きく下がりますが、Sonnetで23件中11件、Opusで29件中12件に当たり、評価品質への影響が大きい可能性があります。初期の本番候補としては厳しすぎます。

## 現時点の推奨

第一候補は `80問 + 1M input cap + 60 tool cap` です。Sonnet/Opus級では実質的に1M input capが効き、tool capは異常ループの保険になります。コストはSonnet級で約$179、Opus/GPT-5.5級で約$291の見込みです。

さらにコストを下げる必要がある場合は、`80問 + 500k input cap + 60 tool cap` より先に、`Compact80 + 1M input cap + 60 tool cap` を検討する方がよいです。500k capは上位モデルの識別力を削るリスクが高く、Compact80は問題選定で代表性を管理しやすいためです。

`Compact40 + 1M input cap + 40 tool cap` は最終手段です。費用は下がりますが、サンプル数を削りすぎるため、リーダーボードの分解能と市場評価との整合性が弱くなります。
