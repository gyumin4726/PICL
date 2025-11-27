CONFIG="config_picl.py"
WORK_DIR="work_dirs/picl_vmamba_base"
GPUS=1
python train_picl.py ${CONFIG} --work-dir ${WORK_DIR} --device cuda
python inference_picl.py ${CONFIG} ${WORK_DIR}/best.pth --mode dataset --output-dir ${WORK_DIR}/inference_results --device cuda
echo "Results saved in: ${WORK_DIR}"