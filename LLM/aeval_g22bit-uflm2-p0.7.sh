#!bin/bash

# ORIG_MODEL=meta-llama/Llama-3.2-1B-Instruct
ORIG_MODEL=google/gemma-2-2b-it
CONFIG_PATH="./config/model/gemma2-2b-it.yaml"
SEARCH_DIR=".cache/ubuntu"
# MODEL=gemma2-2b-it
MODEL_NAME=g22bit
BETA=1.0

RHO=0.7
CP=21
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

RHO=0.7
CP=51
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

RHO=0.7
CP=81
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-${MODEL_NAME}-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED
