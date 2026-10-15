#!/usr/bin/env bash
# 码本规模（--dict_len）扫描：EFBook 在 SEED 数据集上单独调码本大小。
#
# 用法：
#   ./multi_codebook.sh 128 256 512 1024                     # split 模式，SEEDIV SEEDV 各扫一遍
#   ./multi_codebook.sh --mode loso 128 256 512              # LOSO 模式
#   ./multi_codebook.sh --dataset SEEDV 128 256 512          # 只在 SEEDV 上扫
#   ./multi_codebook.sh --fixed-threshold 128 256            # threshold 固定 60，不随 D 缩放
#
# 说明：
#   - 默认 split 模式（Train_Split.py --model EFBook），可用 --mode loso 切换。
#   - threshold（死码字替换窗口）默认随 dict_len 自动缩放，保持每码字期望命中次数一致：
#         threshold = 15 * D / batch_size
#     （D=128 -> 60，D=256 -> 120，与 Args.py 中注释一致；假设 batch_size=32）
#     若想严格只改 dict_len 单变量，用 --fixed-threshold。

set -u

WORK_DIR="/data_main/junlin/MBI/SEED/Main"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"
MODEL="EFBook"
BATCH_SIZE=32
HITS_PER_CODE=15

MODE="split"
DATASETS=(SEEDIV SEEDV)
SIZES=()
FIXED_THRESHOLD=0

while [ $# -gt 0 ]; do
	case "$1" in
		--mode)     MODE="$2"; shift 2 ;;
		--dataset)  DATASETS=("$2"); shift 2 ;;
		--fixed-threshold) FIXED_THRESHOLD=1; shift ;;
		*)          SIZES+=("$1"); shift ;;
	esac
done

case "$MODE" in
	split) TRAIN_SCRIPT="Train_Split.py" ;;
	loso)  TRAIN_SCRIPT="Train_LOSO.py" ;;
	*)     echo "未知 --mode: $MODE（仅支持 split / loso）" >&2; exit 1 ;;
esac

if [ ${#SIZES[@]} -eq 0 ]; then
	echo "用法：$0 [--mode split|loso] [--dataset D] [--fixed-threshold] <codebook_size...>" >&2
	echo "示例：$0 128 256 512 1024" >&2
	exit 1
fi

run_one() {
	local dataset="$1"
	local dict_len="$2"
	local threshold
	if [ "$FIXED_THRESHOLD" -eq 1 ]; then
		threshold=60
	else
		threshold=$(( HITS_PER_CODE * dict_len / BATCH_SIZE ))
	fi
	echo "============================================================"
	echo "SWEEP START: mode=${MODE} dataset=${dataset} dict_len=${dict_len} threshold=${threshold} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD "$TRAIN_SCRIPT" --model "$MODEL" --dataset "$dataset" \
		--dict_len "$dict_len" --threshold "$threshold"
	local code=$?
	echo "SWEEP DONE: mode=${MODE} dataset=${dataset} dict_len=${dict_len} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

for d in "${DATASETS[@]}"; do
	for s in "${SIZES[@]}"; do
		run_one "$d" "$s" || exit 1
	done
done
