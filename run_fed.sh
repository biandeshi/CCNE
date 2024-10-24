PD=data/douban
PREFIX1=online
PREFIX2=offline
TRAINRATIO=0.2
LR=0.001
EPOCHS=300
ALPHA=0.8

python Fedtest.py \
--s_edge ${PD}/${PREFIX1}/raw/edgelist \
--t_edge ${PD}/${PREFIX2}/raw/edgelist \
--gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
--train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
--out_path ${PD}/embeddings \
--dim 128 \
--lr ${LR} \
--epochs ${EPOCHS} \
--alpha ${ALPHA} > output_lr_${LR}_epochs_${EPOCHS}_alpha_${ALPHA}.txt