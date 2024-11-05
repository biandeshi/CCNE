DATASET=twitter1_youtube
PD=data/${DATASET}
PREFIX1=twitter
PREFIX2=youtube
TRAINRATIO=0.9
LR=0.001
EPOCHS=200
ALPHA=1.0
RUNFILE=FedWoNeg

for margin in $(seq 0.1 0.1 1.0);do
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

DIM=16
while ["$DIM" -le "512"];do
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

for alpha in $(seq 0.0 0.1 1.0);do
    python ${RUNFILE}.py \
    --s_edge ${PD}/${PREFIX1}/raw/edgelist \    
    --t_edge ${PD}/${PREFIX2}/raw/edgelist \
    --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
    --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
    --out_path ${PD}/embeddings \
    --dim 128 \
    --lr ${LR} \
    --epochs ${EPOCHS} \
    --alpha ${alpha} > output/${DATASET}/${RUNFILE}_alpha=${alpha}.txt
done

for epochs in $(seq 100 100 1000);do
    python ${RUNFILE}.py \
    --s_edge ${PD}/${PREFIX1}/raw/edgelist \
    --t_edge ${PD}/${PREFIX2}/raw/edgelist \
    --gt_path ${PD}/anchor/node,split=${TRAINRATIO}.test.dict \
    --train_path ${PD}/anchor/node,split=${TRAINRATIO}.train.dict \
    --out_path ${PD}/embeddings \
    --dim 128 \
    --lr ${LR} \
    --epochs ${epochs} \
    --alpha ${ALPHA} > output/${DATASET}/${RUNFILE}_epochs=${epochs}.txt
done