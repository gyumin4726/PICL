#!/bin/bash
# PICL 실행 스크립트
# VMamba Backbone을 이용한 광학 산란 예측

# ===========================
# 설정
# ===========================
CONFIG="config_picl.py"
WORK_DIR="work_dirs/picl_vmamba_base"
GPUS=1

# ===========================
# 학습 (Training)
# ===========================
echo "========================================="
echo "PICL Training"
echo "========================================="
python train_picl.py ${CONFIG} --work-dir ${WORK_DIR} --device cuda

# ===========================
# 추론 (Inference) - 데이터셋 전체
# ===========================
echo ""
echo "========================================="
echo "PICL Inference on Test Dataset"
echo "========================================="
python inference_picl.py ${CONFIG} ${WORK_DIR}/best.pth --mode dataset --output-dir ${WORK_DIR}/inference_results --device cuda

echo ""
echo "✓ PICL pipeline completed!"
echo "Results saved in: ${WORK_DIR}"

