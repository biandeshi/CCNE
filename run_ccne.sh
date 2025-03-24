# PD=data/douban
# PREFIX1=online
# PREFIX2=offline
# TRAINRATIO=0.2

# python CCNE.py \
# --s_edge ${PD}/${PREFIX1}/raw/edgelist \
# --t_edge ${PD}/${PREFIX2}/raw/edgelist \
# --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
# --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
# --out_path ${PD}/embeddings \
# --dim 128 \
# --lr 0.001 \
# --epochs 1000 \
# --margin 0.9

DATASETS=(douban twitter_foursquare twitter1_youtube)
PREFIX1S=(online twitter twitter1)
PREFIX2S=(offline foursquare youtube)
EPOCHS=200
ALPHA=1.0
LR=0.001
RUNFILE=CCNE

mkdir output/${DATASETS[0]}
mkdir output/${DATASETS[1]}
mkdir output/${DATASETS[2]}

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    PREFIX1=${PREFIX1S[$i]}
    PREFIX2=${PREFIX2S[$i]}
    PD=data/${DATASET}

    for TRAINRATIO in $(seq 0.1 0.1 0.9); do

        python ${RUNFILE}.py \
        --s_edge ${PD}/${PREFIX1}/raw/edgelist \
        --t_edge ${PD}/${PREFIX2}/raw/edgelist \
        --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
        --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
        --out_path ${PD}/embeddings \
        --dim 128 \
        --lr 0.001 \
        --epochs 1000 \
        --margin 0.9 > output/${DATASET}/${RUNFILE}_tr=${TRAINRATIO}.txt
    done
done