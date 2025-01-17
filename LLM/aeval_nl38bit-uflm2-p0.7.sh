#!bin/bash

# ORIG_MODEL=meta-llama/Llama-3.2-1B-Instruct
ORIG_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
CONFIG_PATH="./config/model/llama3-8b-it.yaml"
SEARCH_DIR=".cache/ubuntu"
BETA=1.0

RHO=0.7
CP=21
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED

RHO=0.7
CP=51
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED

RHO=0.7
CP=81
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl38bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --config_path1 $CONFIG_PATH \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --use_lora \
    --seed $SEED
