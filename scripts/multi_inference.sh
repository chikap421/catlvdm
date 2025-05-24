#!/bin/bash

CONFIG="configs/t2v_inference_deepspeed.yaml"

PATH_TO_LOG_DIR="/path/to/log_dir"
PATH_TO_CORRUPTION_DIR="/path/to/corruption_dir"

# Study: Varying DDIM steps
TECHNIQUE=("gaussian") # Specify the techniques you want to run
NOISES=("20" "10") # Specify the noise ratios you want to run (20% = 20, 10% = 10)
GUIDANCES=(9.0)

for TECH in "${TECHNIQUE[@]}"; do
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] >>> Running technique: $TECH"
    if [ "$TECH" = "results_2M_train" ]; then
        for GS in "${GUIDANCES[@]}"; do
            OUTPUT_FOLDER="${PATH_TO_LOG_DIR}/msrvtt/${TECH}_guidance${GS}"

            if [ -d "$OUTPUT_FOLDER" ]; then
                echo "[`date '+%Y-%m-%d %H:%M:%S'`] Skipping ${OUTPUT_FOLDER} — already exists."
                continue
            fi

            sed -i -E "s|'resume_checkpoint': '[^']+',|'resume_checkpoint': '${PATH_TO_CORRUPTION_DIR}${TECH}/checkpoints/${TECH}.pth',|g" "$CONFIG"
            sed -i -E "s|'pretrained': '[^']+_motion_encoder\.pth',|'pretrained': '${PATH_TO_CORRUPTION_DIR}${TECH}/checkpoints/${TECH}_motion_encoder.pth',|g" "$CONFIG"
            sed -i -E "s|guide_scale: [0-9.]+|guide_scale: ${GS}|g" "$CONFIG"
            sed -i -E "s|ddim_timesteps: [0-9]+|ddim_timesteps: 50|g" "$CONFIG"
            sed -i -E "s|^(\s*log_dir:\s*\").*(\")|\1${OUTPUT_FOLDER}\2|g" "$CONFIG"

            echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running ${TECH} with guide_scale=${GS}"
            bash scripts/inference_deepspeed.sh
        done
    else
        for NOISE in "${NOISES[@]}"; do
            for GS in "${GUIDANCES[@]}"; do
                OUTPUT_FOLDER="${PATH_TO_LOG_DIR}/msrvtt/${TECH}_${NOISE}_guidance${GS}"

                if [ -d "$OUTPUT_FOLDER" ]; then
                    echo "[`date '+%Y-%m-%d %H:%M:%S'`] Skipping ${OUTPUT_FOLDER} — already exists."
                    continue
                fi

                sed -i -E "s|'resume_checkpoint': '[^']+',|'resume_checkpoint': '${PATH_TO_CORRUPTION_DIR}${TECH}_${NOISE}/checkpoints/${TECH}_${NOISE}.pth',|g" "$CONFIG"
                sed -i -E "s|'pretrained': '[^']+_motion_encoder\.pth',|'pretrained': '${PATH_TO_CORRUPTION_DIR}${TECH}_${NOISE}/checkpoints/${TECH}_${NOISE}_motion_encoder.pth',|g" "$CONFIG"
                sed -i -E "s|guide_scale: [0-9.]+|guide_scale: ${GS}|g" "$CONFIG"
                sed -i -E "s|ddim_timesteps: [0-9]+|ddim_timesteps: 50|g" "$CONFIG"
                sed -i -E "s|^(\s*log_dir:\s*\").*(\")|\1${OUTPUT_FOLDER}\2|g" "$CONFIG"

                echo "[`date '+%Y-%m-%d %H:%M:%S'`] Running ${TECH}_${NOISE} with guide_scale=${GS}"
                bash scripts/inference_deepspeed.sh
            done
        done
    fi
done