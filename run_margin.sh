#!/bin/bash

DATASETS=(douban twitter_foursquare twitter_youtube)
PREFIX1S=(online twitter twitter)
PREFIX2S=(offline foursquare youtube)
EPOCHS=200
ALPHA=1.0
LR=0.001
RUNFILE=FedGAN

mkdir output/${DATASETS[0]}
mkdir output/${DATASETS[1]}
mkdir output/${DATASETS[2]}

DATASET=${DATASETS[2]}
PD=data/${DATASET}
PREFIX1=${PREFIX1S[2]}
PREFIX2=${PREFIX2S[2]}
TRAINRATIO=0.9

for margin in $(seq 0.0 0.1 1.0);do
    python ${RUNFILE}.py \
    --s_edge ${PD}/${PREFIX1}/raw/edgelist \
    --t_edge ${PD}/${PREFIX2}/raw/edgelist \
    --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
    --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
    --out_path ${PD}/embeddings \
    --dim 128 \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --margin ${margin} \
    --alpha ${ALPHA} > output/${DATASET}/${RUNFILE}_margin=${margin}.txt
done
