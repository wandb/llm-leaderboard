import os
import torch
import argparse
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

def count_parameters_from_safetensors(model_path):
    total_parameters = 0
    for filename in os.listdir(model_path):  # モデルディレクトリ内のファイルをループ
        if filename.endswith(".safetensors"):
            filepath = os.path.join(model_path, filename)
            tensor_dict = load_file(filepath)  # safetensors ファイルを読み込み
            for tensor in tensor_dict.values():
                total_parameters += tensor.numel()  # パラメータ数を加算
            del tensor_dict  # メモリ解放
            torch.cuda.empty_cache()
    return total_parameters

def main():
    parser = argparse.ArgumentParser(description='モデルのパラメータ数を計算します。')
    parser.add_argument('model_path', type=str, help='モデルのパスを指定してください。')
    args = parser.parse_args()

    # パラメータ数の計算と出力
    num_parameters = count_parameters_from_safetensors(args.model_path)
    print(f"{args.model_path} のパラメータ数: {num_parameters:,}")

if __name__ == "__main__":
    main()