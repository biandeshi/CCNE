#!/bin/bash

DATASETS=(douban twitter_foursquare twitter1_youtube)
PREFIX1S=(online twitter foursquare)
PREFIX2S=(offline foursquare youtube)
EPOCHS=200
ALPHA=1.0
LR=0.001
RUNFILE=FedWoNeg

mkdir output/${DATASETS[0]}
mkdir output/${DATASETS[1]}
mkdir output/${DATASETS[2]}

DATASET=twitter1_youtube
PD=data/${DATASET}
PREFIX1=twitter1
PREFIX2=youtube
TRAINRATIO=0.9

DIM=16
while [ "$DIM" -le "512" ];do
    python ${RUNFILE}.py \
    --s_edge ${PD}/${PREFIX1}/raw/edgelist \
    --t_edge ${PD}/${PREFIX2}/raw/edgelist \
    --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
    --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
    --out_path ${PD}/embeddings \
    --dim ${DIM} \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --alpha ${ALPHA} > output/${DATASET}/${RUNFILE}_dim=${DIM}.txt
    DIM=$((DIM*2))
done
