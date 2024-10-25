PD=data/twitter1_youtube
PREFIX1=twitter1
PREFIX2=youtube
TRAINRATIO=0.2

python CCNE.py \
--s_edge ${PD}/${PREFIX1}/raw/edgelist \
--t_edge ${PD}/${PREFIX2}/raw/edgelist \
--gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
--train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
--out_path ${PD}/embeddings \
--dim 128 \
--lr 0.001 \
--epochs 1000 \
--margin 0.9