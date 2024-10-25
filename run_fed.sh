DATASET=twitter1_youtube
PD=data/${DATASET}
PREFIX1=twitter1
PREFIX2=youtube
TRAINRATIO=0.2
LR=0.001
EPOCHS=1000
ALPHA=1.0

mkdir output/${DATASET}
python FedWoNeg.py \
--s_edge ${PD}/${PREFIX1}/raw/edgelist \
--t_edge ${PD}/${PREFIX2}/raw/edgelist \
--gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
--train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
--out_path ${PD}/embeddings \
--dim 128 \
--lr ${LR} \
--epochs ${EPOCHS} \
--alpha ${ALPHA} > output/${DATASET}/${PREFIX1}_${PREFIX2}_lr=${LR}_epochs=${EPOCHS}_alpha=${ALPHA}.txt