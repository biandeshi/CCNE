DATASET=douban
PD=data/${DATASET}
PREFIX1=online
PREFIX2=offline
EPOCHS=1000
ALPHA=1.0

for TRAINRATIO in $(seq 0.1 0.1 0.9);do
    for LR in $(seq 0.001 0.001 0.01);do
        for ALPHA in $(seq 0.0 0.1 1.0);do
            python FedWoNeg.py \
            --s_edge ${PD}/${PREFIX1}/raw/edgelist \
            --t_edge ${PD}/${PREFIX2}/raw/edgelist \
            --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
            --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
            --out_path ${PD}/embeddings \
            --dim 128 \
            --lr ${LR} \
            --epochs ${EPOCHS} \
            --alpha ${ALPHA} > output/${DATASET}/${PREFIX1}_${PREFIX2}_tr=${TRAINRATIO}_lr=${LR}_epochs=${EPOCHS}_alpha=${ALPHA}.txt
        done
    done
done