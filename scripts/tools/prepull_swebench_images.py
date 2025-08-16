#!/usr/bin/env python3
"""
SWE-Bench Verified 用の評価イメージを事前取得するユーティリティ

機能:
- .env から DOCKER_USERNAME / DOCKER_PASSWORD を読み取り、Docker Hub にログイン
- Hugging Face の "princeton-nlp/SWE-bench_Verified" から test split を読み込み
- --max-samples の件数だけ instance_id を取得し、公式命名規則に基づくイメージ名を生成
- 並列に docker pull を実行（失敗は記録し最後に要約）

使用例:
  uv run scripts/tools/prepull_swebench_images.py --max-samples 80
  uv run scripts/tools/prepull_swebench_images.py --max-samples 500

注意:
- 非常に大容量です。ディスク容量に注意してください。
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-pull SWE-Bench Verified docker images")
    parser.add_argument(
        "--max-samples",
        type=int,
        choices=[80, 500],
        required=True,
        help="取得するサンプル数（80 または 500）",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="swebench",
        help="レジストリのネームスペース（既定: swebench）",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="イメージタグ（既定: latest）",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="並列 pull 数（既定: 8）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="イメージ名のリスト出力のみを行い、実際の pull は実行しない",
    )
    return parser.parse_args()


def docker_login_from_env() -> None:
    load_dotenv()
    username = os.getenv("DOCKER_USERNAME")
    password = os.getenv("DOCKER_PASSWORD")
    if not username or not password:
        print("[WARN] DOCKER_USERNAME/DOCKER_PASSWORD が .env に見つかりません。匿名での pull を試みます (rate limit の可能性)。")
        return
    try:
        print(f"[INFO] Docker Hub にログイン中: {username}")
        proc = subprocess.run(
            ["docker", "login", "-u", username, "--password-stdin"],
            input=password.encode(),
            capture_output=True,
            check=True,
        )
        print("[INFO] Docker Hub ログイン成功")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Docker Hub ログイン失敗: {e.stderr.decode(errors='ignore').strip()}\n匿名で続行します。")


def ensure_docker_access() -> None:
    """docker CLI の存在とソケット権限を事前チェック。問題があれば明示的に終了。"""
    if shutil.which("docker") is None:
        print("[ERROR] docker コマンドが見つかりません。Docker Engine をインストールしてください。")
        print("        例: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh")
        sys.exit(2)
    # デーモン接続と権限確認
    proc = subprocess.run(["docker", "ps"], capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").lower()
        if "permission denied" in err or "connect: permission denied" in err:
            print("[ERROR] Docker デーモンソケットへのアクセス権限がありません。")
            print("        対処: sudo usermod -aG docker $USER && newgrp docker && docker ps")
            print("        あるいは一時的に sudo -E python3 ... で実行（推奨しません）。")
            sys.exit(2)
        if "is the docker daemon running" in err or "cannot connect to the docker daemon" in err:
            print("[ERROR] Docker デーモンに接続できません。サービスを起動してください。")
            print("        対処: sudo systemctl start docker && docker ps")
            sys.exit(2)
        # その他のエラー
        print(f"[ERROR] docker ps でエラーが発生しました: {proc.stderr or proc.stdout}")
        sys.exit(2)


def load_instance_ids(max_samples: int) -> List[str]:
    try:
        from datasets import load_dataset
    except Exception as e:
        print("[ERROR] datasets が見つかりません。uv 経由で実行してください。例: \n  uv run scripts/tools/prepull_swebench_images.py --max-samples 80")
        raise

    print("[INFO] Hugging Face から SWE-bench Verified (test) を取得中...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    num = min(max_samples, len(ds))
    ids = [str(ds[i]["instance_id"]) for i in range(num)]
    print(f"[INFO] 取得したインスタンス数: {len(ids)}")
    return ids


def to_image_name(namespace: str, instance_id: str, tag: str) -> str:
    # 公式の命名規則に合わせる: __ を _1776_ に置換し、lower()
    formatted = instance_id.replace("__", "_1776_").lower()
    return f"{namespace}/sweb.eval.x86_64.{formatted}:{tag}"


def docker_pull(image: str) -> Tuple[str, bool, str]:
    try:
        proc = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        ok = proc.returncode == 0
        msg = proc.stdout if ok else (proc.stderr or proc.stdout)
        return image, ok, msg.strip()
    except Exception as e:
        return image, False, str(e)


def main() -> int:
    args = parse_args()

    # まず docker アクセスを確認
    ensure_docker_access()

    # 認証は任意（匿名でも可）。失敗しても続行。
    docker_login_from_env()

    instance_ids = load_instance_ids(args.max_samples)
    images = [to_image_name(args.namespace, iid, args.tag) for iid in instance_ids]

    # 保存
    out_dir = Path("./artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / f"swebench_images_{args.max_samples}.txt"
    list_path.write_text("\n".join(images), encoding="utf-8")
    print(f"[INFO] イメージ一覧を書き出しました: {list_path}")

    if args.dry_run:
        print("[DRY-RUN] pull は実行しません。")
        return 0

    print(f"[INFO] docker pull を並列 {args.parallelism} 並列で開始します（{len(images)} 件）...")
    success: List[str] = []
    failed: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.parallelism) as ex:
        futures = {ex.submit(docker_pull, img): img for img in images}
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                image, ok, msg = fut.result()
            except Exception as e:
                ok = False
                msg = str(e)
                image = img
            if ok:
                success.append(image)
                print(f"[OK] {image}")
            else:
                failed.append((image, msg))
                print(f"[FAIL] {image}: {msg}")

    print("\n===== 結果 =====")
    print(f"成功: {len(success)} / 失敗: {len(failed)}")
    if failed:
        fail_path = out_dir / f"swebench_images_failed_{args.max_samples}.txt"
        fail_path.write_text("\n".join([f"{img}\t{msg}" for img, msg in failed]), encoding="utf-8")
        print(f"[INFO] 失敗リスト: {fail_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

