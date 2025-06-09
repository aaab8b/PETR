export CUDA_VISIBLE_DEVICES=5
export HIP_VISIBLE_DEVICES=5
export PYTHONPATH=${PWD}:${PYTHONPATH}
## normal config
CONFIG=${PWD}/projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320_no_ckpt.py 
WORK_DIR=${PWD}/train_full_work_dirs_fp32
## profile config
# CONFIG=${PWD}/projects/configs/maptrv2/maptrv2_nusc_r50_24ep-profile.py 
# WORK_DIR=${PWD}/profiler_logs

## for collect miopen & hipblaslt info
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export HIPBLASLT_LOG_MASK=64
# export HIPBLASLT_LOG_FILE=perf_ana/hipblaslt_mask64_iter5_%i.log
# CONFIG=${PWD}/projects/configs/maptrv2/maptrv2_nusc_r50_24ep-iter.py 
# WORK_DIR=${PWD}/work_dirs
# bash ./tools/bf16/dist_train.sh $CONFIG 8  --work-dir ${WORK_DIR} # 2>&1 | tee perf_ana/miopen_raw_info_1iter.log
tools/dist_train.sh $CONFIG 1 --work-dir ${WORK_DIR}
#WORK_DIR=${PWD}/profiler_logs
#TRACE_JSON=${WORK_DIR}/dl-server-h20_11229.1745496116020.pt.trace.json.gz
#PERF_XLSX=${WORK_DIR}/maptr-v2-batch4-h20.xlsx
#python TraceLens/examples/generate_perf_report.py --profile_path ${TRACE_JSON} --output_xlsx_path ${PERF_XLSX}
