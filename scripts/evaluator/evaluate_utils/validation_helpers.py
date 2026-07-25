"""Validate benchmark output-token allocation before paid evaluation."""

from typing import Optional, Tuple

from omegaconf import DictConfig


BENCHMARK_MIN_OUTPUT_TOKENS = {
    # These evaluators produce long-form answers and must not inherit the short-answer
    # global default. Keep these in sync with the production base configurations.
    "mtbench": 1024,
    "hle": 4096,
}

RUNNER_MANAGED_TOKEN_BUDGETS = {
    "agentic_math",
    "agentic_swe_assorted",
    "deepswe",
    "swebench_pro",
}


def get_reasoning_tokens(cfg: DictConfig) -> Optional[int]:
    """設定からreasoning用のトークン数を取得"""
    try:
        # generator.extra_body.reasoning.max_tokens を確認
        if hasattr(cfg, 'generator') and hasattr(cfg.generator, 'extra_body'):
            extra_body = cfg.generator.extra_body
            if hasattr(extra_body, 'reasoning') and hasattr(extra_body.reasoning, 'max_tokens'):
                return int(extra_body.reasoning.max_tokens)
    except (AttributeError, ValueError, TypeError):
        pass
    
    return None


def get_max_output_tokens(cfg: DictConfig, benchmark_name: str) -> Optional[int]:
    """ベンチマーク用の最大出力トークン数を取得"""
    try:
        # ベンチマーク固有の設定を確認
        benchmark_cfg = getattr(cfg, benchmark_name, None)
        if benchmark_cfg is not None:
            # 各ベンチマークの設定パターンを確認
            patterns = [
                'max_tokens',
                'max_new_token', 
                'max_completion_tokens',
                'max_output_tokens'
            ]
            
            for pattern in patterns:
                if hasattr(benchmark_cfg, pattern):
                    value = getattr(benchmark_cfg, pattern)
                    if value is not None:
                        return int(value)
            
            # generator_config.max_tokensも確認
            if hasattr(benchmark_cfg, 'generator_config'):
                gen_cfg = benchmark_cfg.generator_config
                if hasattr(gen_cfg, 'max_tokens'):
                    return int(gen_cfg.max_tokens)
        
        # フォールバック: generator.max_tokens
        if hasattr(cfg, 'generator') and hasattr(cfg.generator, 'max_tokens'):
            return int(cfg.generator.max_tokens)
            
    except (AttributeError, ValueError, TypeError):
        pass
    
    return None


def check_token_allocation(cfg: DictConfig, benchmark_name: str) -> Tuple[bool, str]:
    """
    トークン配分をチェックして、reasoning後に出力用トークンが確保されているかを確認
    
    Returns:
        (is_valid, message): バリデーション結果とメッセージ
    """
    if benchmark_name in RUNNER_MANAGED_TOKEN_BUDGETS:
        return True, (
            f"✓ {benchmark_name}: OpenClaw runner固有のturn/tool/token予算を"
            "別途検証します（共通generator.max_tokensは未使用）"
        )

    reasoning_tokens = get_reasoning_tokens(cfg)
    max_output_tokens = get_max_output_tokens(cfg, benchmark_name)
    
    # 最大出力トークンが取得できない場合は警告
    if max_output_tokens is None:
        return False, f"⚠️  {benchmark_name}: 最大出力トークン数が設定されていません"

    minimum_output_tokens = BENCHMARK_MIN_OUTPUT_TOKENS.get(benchmark_name)
    if minimum_output_tokens is not None and max_output_tokens < minimum_output_tokens:
        return False, (
            f"❌ {benchmark_name}: 最大出力トークンがベンチマーク最低値を下回っています\n"
            f"   設定値: {max_output_tokens}\n"
            f"   最低値: {minimum_output_tokens}\n"
            "   ベンチマーク固有のgenerator_config.max_tokensを設定してください"
        )
    
    # reasoning機能が使用されていない場合
    if reasoning_tokens is None:
        if max_output_tokens > 0:
            return True, (
                f"✓ {benchmark_name}: 出力トークン数OK ({max_output_tokens}; "
                "Reasoning機能未使用)"
            )
        else:
            return False, f"❌ {benchmark_name}: 最大出力トークンが0以下です ({max_output_tokens})"
    
    # reasoning機能が使用されている場合
    # 実際の出力用トークン = 最大出力トークン - reasoning用トークン
    available_output_tokens = max_output_tokens - reasoning_tokens
    
    if available_output_tokens <= 0:
        return False, (
            f"❌ {benchmark_name}: 出力用トークンが不足しています\n"
            f"   最大出力トークン: {max_output_tokens}\n"
            f"   Reasoning用トークン: {reasoning_tokens}\n"
            f"   出力用残りトークン: {available_output_tokens} (≤ 0)"
        )
    elif available_output_tokens < 512:  # reasoning使用時は512以上推奨
        return False, (
            f"⚠️  {benchmark_name}: Reasoning使用時の出力用トークンが少ないです\n"
            f"   最大出力トークン: {max_output_tokens}\n"
            f"   Reasoning用トークン: {reasoning_tokens}\n"
            f"   出力用残りトークン: {available_output_tokens} (< 512推奨)"
        )
    else:
        return True, (
            f"✓ {benchmark_name}: トークン配分OK\n"
            f"   最大出力トークン: {max_output_tokens}\n"
            f"   Reasoning用トークン: {reasoning_tokens}\n"
            f"   出力用残りトークン: {available_output_tokens}"
        )


def pre_evaluation_check(cfg: DictConfig, benchmark_name: str) -> bool:
    """
    評価実行前のトークン配分チェック
    
    Args:
        cfg: OmegaConf設定オブジェクト
        benchmark_name: ベンチマーク名 (mtbench, bfcl, swebench等)
    
    Returns:
        bool: 評価を続行して良いかどうか
    """
    is_valid, message = check_token_allocation(cfg, benchmark_name)
    
    print("=" * 60)
    print("🔍 トークン配分チェック")
    print("=" * 60)
    print(message)
    print("=" * 60)
    
    if not is_valid:
        print("\n💡 推奨対応:")
        reasoning_tokens = get_reasoning_tokens(cfg)
        if reasoning_tokens:
            print(f"   1. reasoning.max_tokensを{reasoning_tokens}から削減")
            print(f"   2. 各ベンチマークのmax_tokensを増加")
            print(f"   3. 一時的にreasoning機能を無効化")
        print("\n⚠️  このまま評価を続行すると、空白回答により不当に低いスコアになる可能性があります")
        
        # 強制終了はせず、警告のみ表示
        response = input("\n続行しますか？ (y/N): ").strip().lower()
        return response in ['y', 'yes']
    
    return True


def validate_all_benchmarks(cfg: DictConfig) -> dict[str, Tuple[bool, str]]:
    """
    すべてのベンチマークのトークン配分をチェック
    
    Returns:
        Dict[benchmark_name, (is_valid, message)]
    """
    # 主要なベンチマーク一覧
    benchmarks = [
        'mtbench', 'bfcl', 'agentic_math', 'swebench', 'swebench_pro', 'deepswe',
        'agentic_swe_assorted',
        'jbbq', 'toxicity', 'jtruthfulqa', 'hle', 'hallulens',
        'hallulens_zh_tw', 'arc_agi', 'm_ifeval', 'ifeval_zh_tw',
        'ts_bench', 'twbias', 'jaster', 'tceval_v2',
    ]
    
    results = {}
    for benchmark in benchmarks:
        # 実際に設定されているベンチマークのみチェック
        if hasattr(cfg, benchmark):
            results[benchmark] = check_token_allocation(cfg, benchmark)
    
    return results


if __name__ == "__main__":
    # デバッグ用：設定ファイルを直接テスト
    from omegaconf import OmegaConf
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        try:
            cfg = OmegaConf.load(config_path)
            print(f"設定ファイル: {config_path}")
            results = validate_all_benchmarks(cfg)
            
            print("\n" + "=" * 60)
            print("📊 全ベンチマーク トークン配分チェック結果")
            print("=" * 60)
            
            for benchmark, (is_valid, message) in results.items():
                print(f"\n{message}")
                
        except Exception as e:
            print(f"エラー: {e}")
    else:
        print("使用方法: python validation_helpers.py <config_file.yaml>")
