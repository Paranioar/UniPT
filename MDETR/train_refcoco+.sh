CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 11011 --use_env main.py --output_dir runs/unipt_refcoco+ --dataset_config configs/refcoco+.json --batch_size 4 --load pretrained_weights/pretrained_resnet101_checkpoint.pth --ema --text_encoder_lr 1e-5 --lr 5e-4
