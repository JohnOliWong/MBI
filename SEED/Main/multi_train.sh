#!/usr/bin/env bash
# Batch training script for MBI/SEED (SEED-IV / SEED-V)
#
# Usage: first argument selects the training mode (loso | split),
#        followed by grid / pairs as before.
#
#   1) LOSO mode (per-subject leave-one-out):
#      ./multi_train.sh loso grid "models..." "datasets..."
#      ./multi_train.sh loso pairs "model dataset" ...
#
#   2) SPLIT mode (single-run train/val/test holdout):
#      ./multi_train.sh split grid "models..." "datasets..."
#      ./multi_train.sh split pairs "model dataset" ...
#
# Examples:
#   ./multi_train.sh split grid "EEGNet Conformer EyeFormer CAFNet EFNet VigilanceNet TSMMF" "SEEDIV SEEDV"
#   ./multi_train.sh split grid "EFBook" "SEEDIV SEEDV"
#   ./multi_train.sh split pairs "CAFNet SEEDIV" "EFNet SEEDV"

set -u

WORK_DIR="/data_main/junlin/MBI/SEED/Main"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"

usage() {
	echo 'usage:'
	echo '  ./multi_train.sh loso  grid  "model1 model2 ..." "dataset1 dataset2 ..."'
	echo '  ./multi_train.sh split grid  "model1 model2 ..." "dataset1 dataset2 ..."'
	echo '  ./multi_train.sh loso  pairs "model1 dataset1" "model2 dataset2" ...'
	echo '  ./multi_train.sh split pairs "model1 dataset1" "model2 dataset2" ...'
	exit 1
}

# 第一个参数选择训练模式：loso -> Train_LOSO.py ; split -> Train_Split.py
case "${1:-}" in
	loso)  MODE='loso';  TRAIN_SCRIPT='Train_LOSO.py';  shift ;;
	split) MODE='split'; TRAIN_SCRIPT='Train_Split.py'; shift ;;
	*)     usage ;;
esac

run_one() {
	local model="$1"
	local dataset="$2"
	echo "============================================================"
	echo "TRAIN START: mode=${MODE} model=${model} dataset=${dataset} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD "$TRAIN_SCRIPT" --model "$model" --dataset "$dataset"
	local code=$?
	echo "TRAIN DONE: model=${model} dataset=${dataset} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

case "${1:-}" in
	grid)
		shift
		if [ $# -lt 2 ]; then
			usage
		fi
		MODELS=($1)
		DATASETS=($2)
		for m in "${MODELS[@]}"; do
			for d in "${DATASETS[@]}"; do
				run_one "$m" "$d"
			done
		done
		;;
	pairs)
		shift
		if [ $# -lt 1 ]; then
			usage
		fi
		for pair in "$@"; do
			run_one $pair
		done
		;;
	*)
		usage
		;;
esac

echo "ALL TRAINING JOBS FINISHED @ $(date '+%F %T')"
