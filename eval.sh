#!/usr/bin/env bash
# ============================================================================
# memoria 一键评测脚本
# 用法:
#   ./eval.sh                  # 纯词法维度评测
#   ./eval.sh local            # 本地 BGE 混合维度评测
#   ./eval.sh full             # LoCoMo 完整基准（纯词法）
#   ./eval.sh full local       # LoCoMo 完整基准（本地 BGE 混合）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 确保 venv 存在
if [ ! -f .venv/bin/python ]; then
    echo "❌ .venv 不存在，请先创建: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
    exit 1
fi

export MEMORIA_AUTH_SCHEME=none

run_dimension_eval() {
    local backend="$1"
    local label="$2"
    echo "───────────────────────────────────────────────"
    echo "  维度评测: $label"
    echo "───────────────────────────────────────────────"
    rm -rf .dimension-eval-cache
    if [ "$backend" = "none" ]; then
        .venv/bin/memoria-dimension-eval
    else
        .venv/bin/memoria-dimension-eval --embedding-backend "$backend"
    fi
    echo ""
}

run_full_benchmark() {
    local backend="$1"
    local label="$2"
    echo "───────────────────────────────────────────────"
    echo "  LoCoMo 完整基准: $label"
    echo "───────────────────────────────────────────────"
    rm -rf /tmp/memoria_bench/db
    if [ "$backend" = "none" ]; then
        .venv/bin/memoria-load --data /tmp/memoria_bench/adds.jsonl --data-dir /tmp/memoria_bench/db > /dev/null 2>&1
        .venv/bin/memoria-evaluate --data /tmp/memoria_bench/eval.jsonl --data-dir /tmp/memoria_bench/db
    else
        MEMORIA_EMBEDDING_BACKEND="$backend" .venv/bin/memoria-load --data /tmp/memoria_bench/adds.jsonl --data-dir /tmp/memoria_bench/db > /dev/null 2>&1
        MEMORIA_EMBEDDING_BACKEND="$backend" .venv/bin/memoria-evaluate --data /tmp/memoria_bench/eval.jsonl --data-dir /tmp/memoria_bench/db
    fi
    echo ""
}

# ── 解析参数 ──────────────────────────────────────────────────────────────
MODE="${1:-dimension}"   # dimension | full
BACKEND="${2:-none}"     # none | local

if [ "$MODE" = "dimension" ]; then
    run_dimension_eval "$BACKEND" "$BACKEND"
elif [ "$MODE" = "full" ]; then
    # 检查 LoCoMo 数据是否已准备
    if [ ! -f /tmp/memoria_bench/adds.jsonl ]; then
        echo "⚠️  LoCoMo 数据未准备，正在下载..."
        if [ ! -d /tmp/LoCoMo_refined ]; then
            git clone --depth 1 https://github.com/mem-eval-suite/LoCoMo_refined.git /tmp/LoCoMo_refined
        fi
        mkdir -p /tmp/memoria_bench
        .venv/bin/memoria-prepare-benchmark \
            --benchmark locomo_refined \
            --input /tmp/LoCoMo_refined/data/public/conversations.jsonl \
            --questions /tmp/LoCoMo_refined/data/public/questions.jsonl \
            --adds-out /tmp/memoria_bench/adds.jsonl \
            --eval-out /tmp/memoria_bench/eval.jsonl
    fi
    run_full_benchmark "$BACKEND" "$BACKEND"
else
    echo "用法: $0 [dimension|full] [none|local]"
    exit 1
fi