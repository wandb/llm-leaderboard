#!/bin/bash
set -e

# Dockerソケットのグループ権限を取得
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo "")

if [ -n "$DOCKER_GID" ] && [ -S /var/run/docker.sock ]; then
    echo "Docker socket found with GID: $DOCKER_GID"
    
    # dockerグループが存在しない場合は作成
    if ! getent group docker > /dev/null 2>&1; then
        echo "Creating docker group with GID: $DOCKER_GID"
        groupadd -g $DOCKER_GID docker
    fi
    
    # 現在のユーザーをdockerグループに追加
    if [ -n "$USER" ] && [ "$USER" != "root" ]; then
        echo "Adding user $USER to docker group"
        usermod -aG docker $USER || true
    fi
    
    # rootユーザーでもdockerグループに追加（念のため）
    usermod -aG docker root || true
fi

# uvのPATHを確実に通す
export PATH="/workspace/.venv/bin:$PATH"

# 元のコマンドを実行
exec "$@" 