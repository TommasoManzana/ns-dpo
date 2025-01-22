#!bin/bash

RHODIFF=0.0
CP=0
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=1.0
CP=81
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=1.0
CP=51
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=1.0
CP=21
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=0.7
CP=81
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=0.7
CP=51
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 

RHODIFF=0.7
CP=21
SEED=2024

python3 src/datasets/create_train_nuf.py \
    --rhodiff $RHODIFF \
    --cp $CP \
    --seed $SEED 