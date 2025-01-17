#!bin/bash

ORIG_MODEL=meta-llama/Llama-3.2-1B-Instruct
SEARCH_DIR=".cache/ubuntu"
BETA=1.0

# RHO=0.0
# CP=0
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# RHO=1.0
# CP=21
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# RHO=1.0
# CP=51
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# RHO=1.0
# CP=66
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# RHO=1.0
# CP=81
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED


# RHO=0.7
# CP=21
# SEED=2028

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

# NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

# DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
# echo "MODEL1 dir", $DIR_MODEL

# python3 test_alpacaeval.py \
#     --original_model $ORIG_MODEL \
#     --name_model $NAME_MODEL \
#     --path_model $DIR_MODEL \
#     --seed $SEED

RHO=0.7
CP=51
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

RHO=0.7
CP=66
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

RHO=0.7
CP=81
SEED=2025

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-dpo-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED

NAME_MODEL=uflm2-rho${RHO}-cp${CP}-nl31bit-nsdpo-g0.85-b${BETA}

DIR_MODEL=$(find $SEARCH_DIR -type d -name $NAME_MODEL* | sort | tail -n 1)
echo "MODEL1 dir", $DIR_MODEL

python3 test_alpacaeval.py \
    --original_model $ORIG_MODEL \
    --name_model $NAME_MODEL \
    --path_model $DIR_MODEL \
    --seed $SEED