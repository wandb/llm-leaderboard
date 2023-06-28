# Weights & BiasesのLaunchを用いて、Hugging Face上のモデルを日本語タスクJGLEUに対して評価！

このgit repositoryでは、Hugging Face上のモデルを日本語タスクJGLEUに対して精度評価するためのscriptを管理しています。
LLMの進展が激しいですが、様々なLLMに対して日本語のタスクに対する精度がはどうなっているのかを検証するといったプロジェクトをWeights＆Biases（以下WB）が進めました。
下記がそのプロジェクトのレポートです。
[LLMのJGLUEによる日本語タスクベンチマーク](https://wandb.ai/wandb/LLM_evaluation_Japan/reports/LLM-JGLUE---Vmlldzo0NTUzMDE2 "LLMのJGLUEによる日本語タスクベンチマーク")

誰でも自身の環境で、上記のレポート内で行った評価を行うことができるように、WBがLaucnhを公開しました！

## Launchとは？
W&B Launchは、ML開発者がモダンなMLワークフローを支える高スケールで専門的なハードウェアをシームレスに使用することを可能にし、学習のスケールアップやモデル評価フローの構築、推論のためのモデル読み込みなどの煩わしさを解消してくれます。



## W＆BのLaunchを用いて、Hugging Face上のモデルを日本語タスクJGLEUに対して評価する方法
下記のステップで日本語タスクを実行することができます。Launchの細かい使い方については、W&Bの[Doc](https://docs.wandb.ai/ja/guides/launch#docusaurus_skipToContent_fallback)を参考にしてください。
0. すでにジョブは作成されています！今回のジョブは下記のpageで見つけることができます。
    W&B Project: [LLM_evaluation_Japan_publicのjobページ](https://wandb.ai/wandb/LLM_evaluation_Japan_public/jobs)
1. [こちら](https://docs.wandb.ai/ja/guides/launch/run-agent)を参考に、エージェントを作成し、エージェントを開始してください。
2. ジョブのページから"Launch"のボタンを押し、キューを作成してください。キューを実行する際に、overridesの中のargsを変更してください。run_configの中の変数は変更する必要はありません。
    * --model_name" : Hugging Face上のモデルの名前を指定してください
    * --prompt_type (alpaca, rinna, pythia, others) : 使用するプロンプトを選択してください。プロンプトの詳細は、src/prompt_template.pyを参考にしてください。現状のversionでは下記の4つのパターンがあります。

## 注意




