#!/bin/bash

# Ensure Git LFS is installed
git lfs install

# Download the base ModelScope text-to-video model
git clone https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis ./modelscopet2v

# Optional: Download CAT-LVDM checkpoints (multiple corruption types and noise levels)
# Uncomment the line below if you wish to download all trained corruption-aware models
# git clone https://huggingface.co/Chikap421/catlvdm-checkpoints ./catlvdm