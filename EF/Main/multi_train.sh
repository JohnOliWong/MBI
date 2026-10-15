#!/usr/bin/env bash
# Batch training script for MBI/EF
#
# Usage:
#   1) grid:  train all combinations of models x datasets
#      ./multi_train.sh grid "EEGNet Conformer fNIRST fNIRSNet CAFNet EFNet VigilanceNet TSMMF" "EFMI EFMA EFWG"
#      ./multi_train.sh grid "EFBook" "EFMI EFMA EFWG"
#      ./multi_train.sh grid "VigilanceNet" "EFMI EFMA"
#   2) pairs: train explicitly specified model-dataset pairs
#      ./multi_train.sh pairs "EEGNet EFMI"

set -u

WORK_DIR="/data_main/junlin/MBI/EF/Main"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"

run_one() {
	local model="$1"
	local dataset="$2"
	echo "============================================================"
	echo "TRAIN START: model=${model} dataset=${dataset} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD Train.py --model "$model" --dataset "$dataset"
	local code=$?
	echo "TRAIN DONE: model=${model} dataset=${dataset} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

case "${1:-}" in
	grid)
		shift
		if [ $# -lt 2 ]; then
			echo 'usage: ./run_train.sh grid "model1 model2 ..." "dataset1 dataset2 ..."'
			exit 1
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
			echo 'usage: ./run_train.sh pairs "model1 dataset1" "model2 dataset2" ...'
			exit 1
		fi
		for pair in "$@"; do
			run_one $pair
		done
		;;
	*)
		echo 'usage:'
		echo '  ./run_train.sh grid "model1 model2 ..." "dataset1 dataset2 ..."'
		echo '  ./run_train.sh pairs "model1 dataset1" "model2 dataset2" ...'
		exit 1
		;;
esac

echo "ALL TRAINING JOBS FINISHED @ $(date '+%F %T')"
