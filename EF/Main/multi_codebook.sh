#!/usr/bin/env bash
# 码本规模（--dict_len）扫描：EFBook 在 EF 数据集上单独调码本大小。
#
# 用法：
#   ./multi_codebook.sh --threshold 60 512 1024 1680 1800                     # EFMI EFMA EFWG 各扫一遍
#   ./multi_codebook.sh --dataset EFWG 128 256 512           # 只在 EFWG 上扫
#   ./multi_codebook.sh --threshold 60 128 256               # threshold 固定 60，不随 D 缩放
#
# 说明：
#   - EF 只有一个训练脚本 Train.py（逐 subject 的 LOSO）。
#   - threshold（死码字替换窗口）默认随 dict_len 缩放，以 EF 默认 D=1800 -> threshold=60 为基线：
#         threshold = 60 * D / 1800
#     （D=1800 -> 60，D=900 -> 30，D=3600 -> 120）
#     若想严格只改 dict_len 单变量，用 --threshold 固定值。

set -u

WORK_DIR="/data_main/junlin/MBI/EF/Main"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"
MODEL="EFBook"
TRAIN_SCRIPT="Train.py"
BASE_DICT=1800
BASE_THRESHOLD=60

DATASETS=(EFMI EFMA EFWG)
SIZES=()
THRESHOLD_FIXED=""

while [ $# -gt 0 ]; do
	case "$1" in
		--dataset)   DATASETS=("$2"); shift 2 ;;
		--threshold) THRESHOLD_FIXED="$2"; shift 2 ;;
		*)           SIZES+=("$1"); shift ;;
	esac
done

if [ ${#SIZES[@]} -eq 0 ]; then
	echo "用法：$0 [--dataset D] [--threshold T] <codebook_size...>" >&2
	echo "示例：$0 128 256 512 1024" >&2
	exit 1
fi

run_one() {
	local dataset="$1"
	local dict_len="$2"
	local threshold
	if [ -n "$THRESHOLD_FIXED" ]; then
		threshold="$THRESHOLD_FIXED"
	else
		threshold=$(( BASE_THRESHOLD * dict_len / BASE_DICT ))
	fi
	echo "============================================================"
	echo "SWEEP START: dataset=${dataset} dict_len=${dict_len} threshold=${threshold} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD "$TRAIN_SCRIPT" --model "$MODEL" --dataset "$dataset" \
		--dict_len "$dict_len" --threshold "$threshold"
	local code=$?
	echo "SWEEP DONE: dataset=${dataset} dict_len=${dict_len} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

for d in "${DATASETS[@]}"; do
	for s in "${SIZES[@]}"; do
		run_one "$d" "$s" || exit 1
	done
done
