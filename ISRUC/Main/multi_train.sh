#!/usr/bin/env bash
# Batch training script for MBI/ISRUC（睡眠分期 5 分类，holdout 划分）
#
# 用法：每个参数是一个模型（普通模型直接写名字；EFBook 用 "EFBook:mods" 指定模态）
#   ./multi_train.sh EEGNet Conformer MyoCNN EFNet VigilanceNet TSMMF
#   ./multi_train.sh Conformer MyoCNN EFNet VigilanceNet TSMMF
#   ./multi_train.sh "EFBook:eeg" "EFBook:eeg,emg" "EFBook:eeg,emg,eog"
#   ./multi_train.sh EEGNet "EFBook:eeg,emg,eog"
#
# 支持的模型：EEGNet Conformer EFNet VigilanceNet TSMMF MyoCNN EFBook
#   MyoCNN 为 EMG-only 基线（自动使用 --modalities emg）
# EFBook 模态（--modalities）：eeg / eeg,emg / eeg,emg,eog（默认 eeg,emg）

set -u

WORK_DIR="/data_main/junlin/MBI/ISRUC/Main"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"
TRAIN_SCRIPT='Train_Split.py'

run_one() {
	local spec="$1"
	local model="${spec%%:*}"
	local mods="${spec#*:}"
	[ "$mods" = "$spec" ] && mods=""

	echo "============================================================"
	echo "TRAIN START: model=${model} modalities=${mods:-<default>} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	if [ -n "$mods" ]; then
		$PY_CMD "$TRAIN_SCRIPT" --model "$model" --modalities "$mods"
	else
		$PY_CMD "$TRAIN_SCRIPT" --model "$model"
	fi
	local code=$?
	echo "TRAIN DONE: model=${model} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

if [ $# -eq 0 ]; then
	echo 'usage: ./multi_train.sh model [model ...]'
	echo '  MyoCNN 为 EMG-only 基线，直接写名字即可'
	echo '  EFBook 用 "EFBook:mods" 指定模态，如 EFBook:eeg,emg,eog'
	exit 1
fi

for spec in "$@"; do
	run_one "$spec"
done

echo "ALL TRAINING JOBS FINISHED @ $(date '+%F %T')"
