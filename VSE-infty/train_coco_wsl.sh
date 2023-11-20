DATASET_NAME='coco'
DATA_PATH='./data/'${DATASET_NAME}
WEIGHT_PATH='./data/weights'
IMG_ADAPTER='our_tuning'
TXT_ADAPTER='our_tuning'
SAVE_NAME=unipt_${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME}  \
  --logger_name runs/${SAVE_NAME}/log --model_name runs/${SAVE_NAME} \
  --num_epochs 25 --lr_update 15 --learning_rate 5e-4 --workers 20 --log_step 200 \
  --precomp_enc_type backbone --backbone_source wsl \
  --vse_mean_warmup_epochs 1 --backbone_warmup_epochs 0 --embedding_warmup_epochs 100 \
  --input_scale_factor 2.0 --backbone_lr_factor 1 --bert_lr_factor 1 \
  --img_adapter_name ${IMG_ADAPTER} --txt_adapter_name ${TXT_ADAPTER} --batch_size 112 \
  --txt_downsample_D_factor 2 --img_downsample_D_factor 2