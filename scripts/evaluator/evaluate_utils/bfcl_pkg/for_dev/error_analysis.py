import pandas as pd
import json
import openai
from typing import Dict, List, Optional
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ErrorAnalyzer:
    def __init__(self, api_key: str, max_workers: int = 8):
        """OpenAI APIキーで初期化"""
        self.client = openai.OpenAI(api_key=api_key)
        self.max_workers = max_workers
        
    def explain_wrong_answer(self, prompt: str, correct_answer: str, wrong_answer: str) -> str:
        """
        間違った答えについて説明を生成
        
        Args:
            prompt: ユーザーのプロンプト
            correct_answer: 正解
            wrong_answer: 間違った答え
            
        Returns:
            説明文
        """
        explanation_prompt = f"""
以下の問題で間違った答えが出されました。なぜ間違っているのかを簡潔に説明してください。

**問題:**
{prompt}

**正解:**
{correct_answer}

**間違った答え:**
{wrong_answer}

間違っている理由を1-2文で簡潔に説明してください。
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "あなたは問題の解説者です。間違った答えの理由を簡潔に説明してください。"},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
                
        except Exception as e:
            return f"説明生成エラー: {str(e)}"
    
    def _process_single_case(self, case_data: tuple) -> Optional[Dict]:
        """
        単一ケースの処理（並列処理用）
        
        Args:
            case_data: (index, row)のタプル
            
        Returns:
            処理結果またはNone
        """
        idx, row = case_data
        
        try:
            # エラー情報から間違った答えを抽出
            try:
                error_data = json.loads(row['error']) if row['error'] != "null" else None
                wrong_answer = error_data.get('model_response', '不明') if error_data else '不明'
            except:
                wrong_answer = row['error']
            
            # 説明を生成
            explanation = self.explain_wrong_answer(
                prompt=row['prompt'],
                correct_answer=row['possible_answer'],
                wrong_answer=wrong_answer
            )
            
            result = {
                "index": idx + 1,
                "id": row.get('id', f"unknown_{idx+1}"),
                "prompt": row['prompt'],
                "correct_answer": row['possible_answer'],
                "wrong_answer": wrong_answer,
                "explanation": explanation
            }
            
            print(f"✓ 完了: {idx+1}")
            return result
            
        except Exception as e:
            print(f"✗ エラー (行 {idx+1}): {str(e)}")
            return None
    
    def analyze_csv_file(self, csv_path: str, output_path: str = None) -> List[Dict]:
        """
        CSVファイルの間違った答えを分析（並列処理版）
        
        Args:
            csv_path: 入力CSVファイルのパス
            output_path: 出力JSONファイルのパス（オプション）
            
        Returns:
            分析結果のリスト
        """
        # CSVファイルを読み込み
        df = pd.read_csv(csv_path)
        
        print(f"CSVファイルを読み込みました。{len(df)}件のデータがあります。")
        print(f"並列処理ワーカー数: {self.max_workers}")
        
        # エラーがあるケースのみを抽出
        error_cases = []
        for idx, row in df.iterrows():
            if pd.notna(row['error']) and row['error'] != "null":
                error_cases.append((idx, row))
        
        print(f"分析対象ケース数: {len(error_cases)}")
        
        results = []
        start_time = time.time()
        
        # 並列処理で分析を実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクを送信
            future_to_idx = {
                executor.submit(self._process_single_case, case): case[0] 
                for case in error_cases
            }
            
            # 完了したタスクを処理
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                if result:
                    results.append(result)
                
                completed += 1
                if completed % 5 == 0:  # 5件ごとに進捗表示
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"進捗: {completed}/{len(error_cases)} ({rate:.2f}件/秒)")
        
        elapsed = time.time() - start_time
        print(f"分析完了: {len(results)}件の説明を生成しました (所要時間: {elapsed:.2f}秒)")
        
        # 結果を保存
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"分析結果を {output_path} に保存しました。")
        
        return results


def main():
    """メイン実行関数"""
    # OpenAI APIキーを環境変数から取得
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("エラー: OPENAI_API_KEY環境変数が設定されていません。")
        return
    
    # 並列処理ワーカー数を設定（環境変数から取得可能）
    max_workers = int(os.getenv('MAX_WORKERS', '8'))
    
    # アナライザーを初期化
    analyzer = ErrorAnalyzer(api_key, max_workers=max_workers)
    
    # スクリプトのディレクトリを基準にCSVファイルのパスを設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "wandb_export_2025-08-07T16_11_32.221+09_00.csv")
    
    results = analyzer.analyze_csv_file(
        csv_path=csv_path,
        output_path="wrong_answer_explanations.json"
    )
    
    # 結果を表示
    print("\n=== 間違った答えの説明 ===")
    for result in results:
        print(f"\n【問題 ID: {result['id']}】")
        print(f"問題: {result['prompt'][:100]}...")
        print(f"正解: {result['correct_answer']}")
        print(f"間違い: {result['wrong_answer']}")
        print(f"説明: {result['explanation']}")
        print("-" * 50)
    
    # 結果をファイルに保存（テキスト形式）
    text_output_path = "wrong_answer_explanations.txt"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write("=== 間違った答えの説明 ===\n\n")
        for result in results:
            f.write(f"【問題 ID: {result['id']}】\n")
            f.write(f"問題: {result['prompt']}\n")
            f.write(f"正解: {result['correct_answer']}\n")
            f.write(f"間違い: {result['wrong_answer']}\n")
            f.write(f"説明: {result['explanation']}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"\n結果を以下のファイルに保存しました:")
    print(f"- JSON形式: wrong_answer_explanations.json")
    print(f"- テキスト形式: {text_output_path}")


if __name__ == "__main__":
    main() 