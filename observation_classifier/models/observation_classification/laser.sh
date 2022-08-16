#!/bin/bash

MAIN_DIR="/home/THISTED-NET/ep/Projects/Text_Classification/Thisted_Kommune/Thisted-Systematic" 
PROJ_DIR="$MAIN_DIR/thisted_observation_classification/observation_classifier/models"
SHARED_DIR="$MAIN_DIR/Text_Classification"
DATA_DIR="$SHARED_DIR/data"
OUTPUT_DIR="$SHARED_DIR/output/models"
script="$PROJ_DIR/main.py"

params="""
  --cfg $PROJ_DIR/laser_XGBoost.yml \
  --gpu_id 0 
  --data_dir $DATA_DIR/Observation/all_data/v4 \
  --output_dir $OUTPUT_DIR \
  --dataset_name random_ObservationData_Combined_QA_comments_embeddings 
  --manualSeed 1
  --embd_dir $DATA_DIR/embeddings \
"""


# Activate environment
source activate environment 

python $script $params > laser.out
 
# Deactivate environment
conda deactivate
