#!/usr/bin/env bash
# 批量运行 EFBook 消融实验
#
# 模块消融（OT / DECOUPLE / PC / SC）：python Ablation_Module.py --dataset <D> --mode <M>
# 单模态（EEG / fNIRS）：              python Ablation_Modality.py --dataset <D> --modality <m>
#
# 用法：
#   ./multi_ablate.sh               # 在 EFMI EFMA EFWG 三个数据集上跑全部 6 种消融
#   ./multi_ablate.sh EFWG          # 只在 EFWG 上跑全部 6 种消融
#   ./multi_ablate.sh EFMI EFWG     # 在 EFMI、EFWG 上跑全部 6 种消融

set -u

WORK_DIR="/data_main/junlin/MBI/EF/Ablation"
PY_CMD="conda run -p /data_main/junlin/conda_env/mbi --no-capture-output python"

# DATA_MODULE=(EFMI EFWG)
# MODULE_MODES=(OT DECOUPLE PC SC)
# DATA_MODALITY=(EFMI)
# MODALITIES=(eeg nirs)

DATA_MODULE=(EFMI)
MODULE_MODES=(OT DECOUPLE)

run_module() {
	local dataset="$1"
	local mode="$2"
	echo "============================================================"
	echo "ABLATE MODULE START: mode=${mode} dataset=${dataset} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD Ablation_Module.py --dataset "$dataset" --mode "$mode"
	local code=$?
	echo "ABLATE MODULE DONE: mode=${mode} dataset=${dataset} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

run_modality() {
	local dataset="$1"
	local modality="$2"
	echo "============================================================"
	echo "ABLATE MODALITY START: modality=${modality} dataset=${dataset} @ $(date '+%F %T')"
	cd "$WORK_DIR" || return 1
	$PY_CMD Ablation_Modality.py --dataset "$dataset" --modality "$modality"
	local code=$?
	echo "ABLATE MODALITY DONE: modality=${modality} dataset=${dataset} exit=${code} @ $(date '+%F %T')"
	return "$code"
}

for d in "${DATA_MODULE[@]}"; do
	for m in "${MODULE_MODES[@]}"; do
		run_module "$d" "$m"
	done
done

# for d in "${DATA_MODALITY[@]}"; do
# 	for mod in "${MODALITIES[@]}"; do
# 		run_modality "$d" "$mod"
# 	done
# done

echo "ALL ABLATION JOBS FINISHED @ $(date '+%F %T')"
