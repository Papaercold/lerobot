#!/bin/bash
# =============================================================================
# SARM Subtask Annotation for Folding Task
# Step 1: Use Qwen3-VL to annotate subtask boundaries in each episode
#
# Usage:
#   chmod +x scripts/annotate_folding.sh
#   ./scripts/annotate_folding.sh
# =============================================================================

set -e  # Exit immediately on error

# =============================================================================
# 配置区域 - 根据你的实际情况修改这里
# =============================================================================

# 你的 LeRobot 数据集（HuggingFace repo ID 或本地路径）
DATASET_REPO_ID="your-username/your-folding-dataset"

# 摄像头视角的 key（用 python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds=LeRobotDataset('$DATASET_REPO_ID'); print(ds.meta.video_keys)" 查看）
VIDEO_KEY="observation.images.top"

# 子任务阶段描述（按执行顺序，逗号分隔）
DENSE_SUBTASKS="reach to shirt, fold left sleeve, fold right sleeve, fold shirt in half"

# VLM 模型（默认 Qwen3-VL-30B，需要约 60GB VRAM）
MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"

# 使用哪张 GPU
DEVICE="cuda"
DTYPE="bfloat16"

# 可视化输出目录（标注完后自动生成检查图）
VIZ_DIR="./outputs/annotation_viz"

# 是否上传标注结果到 HuggingFace Hub（true/false）
PUSH_TO_HUB=false
OUTPUT_REPO_ID="${DATASET_REPO_ID}"  # 上传到哪个 repo（默认同名）

# =============================================================================
# 多 GPU 并行（可选，加速大数据集标注）
# =============================================================================

# 使用几个 worker（每个 worker 占一张 GPU）
NUM_WORKERS=1

# 指定 GPU ID（空表示自动分配）
# 例如两张卡并行: GPU_IDS="0 1"
GPU_IDS=""

# =============================================================================
# 以下无需修改
# =============================================================================

echo "=============================================="
echo " SARM Subtask Annotation - Folding Task"
echo "=============================================="
echo " Dataset:    ${DATASET_REPO_ID}"
echo " Video key:  ${VIDEO_KEY}"
echo " Subtasks:   ${DENSE_SUBTASKS}"
echo " Model:      ${MODEL}"
echo " Workers:    ${NUM_WORKERS}"
echo "=============================================="
echo ""

# 构建命令
CMD="python src/lerobot/data_processing/sarm_annotations/subtask_annotation.py"
CMD="${CMD} --repo-id ${DATASET_REPO_ID}"
CMD="${CMD} --dense-subtasks \"${DENSE_SUBTASKS}\""
CMD="${CMD} --dense-only"
CMD="${CMD} --video-key ${VIDEO_KEY}"
CMD="${CMD} --model ${MODEL}"
CMD="${CMD} --dtype ${DTYPE}"
CMD="${CMD} --num-workers ${NUM_WORKERS}"
CMD="${CMD} --skip-existing"

# 可视化选项
CMD="${CMD} --num-visualizations 5"
CMD="${CMD} --output-dir ${VIZ_DIR}"

# 多 GPU
if [ -n "${GPU_IDS}" ]; then
    CMD="${CMD} --gpu-ids ${GPU_IDS}"
fi

# 上传选项
if [ "${PUSH_TO_HUB}" = "true" ]; then
    CMD="${CMD} --push-to-hub"
    if [ -n "${OUTPUT_REPO_ID}" ]; then
        CMD="${CMD} --output-repo-id ${OUTPUT_REPO_ID}"
    fi
fi

echo "Running command:"
echo "${CMD}"
echo ""

# 执行标注
eval ${CMD}

echo ""
echo "=============================================="
echo " Annotation complete!"
echo " Check visualizations at: ${VIZ_DIR}"
echo ""
echo " Next step (after verifying annotation quality):"
echo ""
echo "   python src/lerobot/scripts/lerobot_train.py \\"
echo "     --policy.type=sarm \\"
echo "     --dataset.repo_id=${DATASET_REPO_ID} \\"
echo "     --policy.annotation_mode=dense_only \\"
echo "     --output_dir=outputs/train/sarm_folding"
echo "=============================================="
