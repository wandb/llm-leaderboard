#!/usr/bin/env python3
"""
JTruthfulQA Scoring Debug Script
JTruthfulQAのスコアリングが止まる問題を診断するためのスクリプト
"""

import os
import sys
import time
import torch
import psutil
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import subprocess
import signal

def monitor_resources():
    """リソース使用量を監視"""
    while True:
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # メモリ使用量
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU情報
            gpu_info = "N/A"
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                gpu_percent = (gpu_used / gpu_memory) * 100
                gpu_info = f"{gpu_percent:.1f}%"
            
            print(f"[MONITOR] CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}% | GPU: {gpu_info}")
            time.sleep(10)
        except Exception as e:
            print(f"[MONITOR ERROR] {e}")
            break

def test_jumanpp_direct():
    """Juman++の直接テスト"""
    print("=== Testing Juman++ directly ===")
    try:
        # 環境変数確認
        jumanpp_cmd = os.environ.get('JUMANPP_COMMAND', '/usr/local/bin/jumanpp')
        print(f"JUMANPP_COMMAND: {jumanpp_cmd}")
        
        # 直接実行テスト
        result = subprocess.run([jumanpp_cmd, '--version'], 
                               capture_output=True, text=True, timeout=10)
        print(f"Version check: {result.returncode}")
        print(f"Output: {result.stdout.strip()}")
        
        # 実際のテキスト処理テスト
        test_text = "これはテストです。"
        process = subprocess.Popen([jumanpp_cmd], 
                                  stdin=subprocess.PIPE, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        stdout, stderr = process.communicate(input=test_text, timeout=10)
        print(f"Processing test: {process.returncode}")
        print(f"Output lines: {len(stdout.split())}")
        
    except subprocess.TimeoutExpired:
        print("❌ Juman++ process timeout!")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Juman++ error: {e}")
        return False
    
    print("✅ Juman++ working normally")
    return True

def test_roberta_tokenizer():
    """RoBERTaトークナイザーのテスト"""
    print("=== Testing RoBERTa Tokenizer ===")
    try:
        # トークナイザー初期化
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/roberta_jtruthfulqa')
        print("✅ Tokenizer loaded")
        
        # テストデータ
        test_cases = [
            "これは短いテストです。",
            "これはより長いテストです。日本語の複雑な文章を処理できるかテストしています。",
            "質問: これは真実ですか？ 回答: はい、これは事実です。",
        ]
        
        for i, text in enumerate(test_cases):
            print(f"Test case {i+1}: {text[:30]}...")
            start_time = time.time()
            
            # タイムアウト付きでトークン化
            tokens = tokenizer.tokenize(text)
            elapsed = time.time() - start_time
            
            print(f"  Tokens: {len(tokens)}, Time: {elapsed:.3f}s")
            if elapsed > 5.0:
                print(f"  ⚠️  Slow tokenization: {elapsed:.3f}s")
            
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False
    
    print("✅ Tokenizer working normally")
    return True

def test_roberta_model():
    """RoBERTaモデルのテスト"""
    print("=== Testing RoBERTa Model ===")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # GPU使用状況チェック
        if torch.cuda.is_available():
            print(f"GPU Memory before model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        
        # モデル読み込み
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/roberta_jtruthfulqa')
        model = AutoModelForSequenceClassification.from_pretrained('nlp-waseda/roberta_jtruthfulqa').to(device)
        
        if torch.cuda.is_available():
            print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        
        # 推論テスト
        test_qa_pairs = [
            ("これは質問ですか？", "はい、これは質問です。"),
            ("2+2は4ですか？", "はい、2+2は4です。"),
            ("地球は平らですか？", "いいえ、地球は球体です。"),
        ]
        
        for i, (question, answer) in enumerate(test_qa_pairs):
            print(f"Test inference {i+1}...")
            start_time = time.time()
            
            # 推論実行
            inputs = tokenizer(question + " " + answer, return_tensors="pt", 
                             truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score = probabilities[0][1].item()
            elapsed = time.time() - start_time
            
            print(f"  Score: {score:.4f}, Time: {elapsed:.3f}s")
            if elapsed > 10.0:
                print(f"  ⚠️  Slow inference: {elapsed:.3f}s")
        
        # クリーンアップ
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False
    
    print("✅ Model working normally")
    return True

def test_batch_processing():
    """バッチ処理のストレステスト"""
    print("=== Testing Batch Processing ===")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/roberta_jtruthfulqa')
        model = AutoModelForSequenceClassification.from_pretrained('nlp-waseda/roberta_jtruthfulqa').to(device)
        
        # 100個のテストケースを生成
        test_questions = [f"これは質問{i}ですか？" for i in range(100)]
        test_answers = [f"はい、これは回答{i}です。" for i in range(100)]
        
        print(f"Processing {len(test_questions)} samples...")
        start_time = time.time()
        processed = 0
        
        for i, (question, answer) in enumerate(zip(test_questions, test_answers)):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i}/100, Rate: {rate:.2f} samples/sec")
            
            inputs = tokenizer(question + " " + answer, return_tensors="pt", 
                             truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            processed += 1
            
            # 異常に時間がかかっている場合は警告
            if i > 0 and (time.time() - start_time) / i > 5.0:
                print(f"  ⚠️  Slow processing detected: {(time.time() - start_time) / i:.3f}s per sample")
                break
        
        total_time = time.time() - start_time
        final_rate = processed / total_time
        print(f"Completed {processed} samples in {total_time:.2f}s ({final_rate:.2f} samples/sec)")
        
        # クリーンアップ
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return False
    
    print("✅ Batch processing completed")
    return True

def main():
    print("JTruthfulQA Scoring Debug Script")
    print("=" * 50)
    
    # リソース監視を別スレッドで開始
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # 各テストを実行
    tests = [
        ("Juman++ Direct Test", test_jumanpp_direct),
        ("RoBERTa Tokenizer Test", test_roberta_tokenizer),
        ("RoBERTa Model Test", test_roberta_model),
        ("Batch Processing Test", test_batch_processing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n❌ {test_name} interrupted by user")
            results[test_name] = False
            break
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # 結果サマリー
    print(f"\n{'='*20} Results Summary {'='*20}")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! JTruthfulQA should work normally.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 