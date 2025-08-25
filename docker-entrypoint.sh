#!/bin/bash
set -e

# Dockerソケットのグループ権限を取得
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo "")

if [ -n "$DOCKER_GID" ] && [ -S /var/run/docker.sock ]; then
    echo "Docker socket found with GID: $DOCKER_GID"
    
    # dockerグループが存在しない場合は作成
    if ! getent group docker > /dev/null 2>&1; then
        echo "Creating docker group with GID: $DOCKER_GID"
        groupadd -g $DOCKER_GID docker || true
    fi
    
    # 現在のユーザーをdockerグループに追加（ルート権限が必要な場合はスキップ）
    if [ -w /etc/passwd ]; then
        if [ -n "$USER" ] && [ "$USER" != "root" ]; then
            echo "Adding user $USER to docker group"
            usermod -aG docker $USER || true
        fi
        
        # rootユーザーでもdockerグループに追加（念のため）
        usermod -aG docker root || true
    else
        echo "Warning: Cannot modify user groups (read-only /etc/passwd)"
        echo "Running with current permissions"
    fi
fi

# uvのPATHを確実に通す
export PATH="/workspace/.venv/bin:$PATH"

# 元のコマンドを実行
exec "$@" 