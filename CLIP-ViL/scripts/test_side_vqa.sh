# The name of this experiment.
r=2
seed=9595
# Save logs and models under snap/vqa; make backup.
output=$2

# export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
    --distributed \
    --train train,nominival --valid minival \
    --test test \
    --load $output/BEST \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 300 --optim bert --lr 5e-4 --epochs 5 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --loss_scale 500 \
    --warmup_ratio 0.05 \
    --report_step 400 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.0001 \
    --sgd_momentum 0.0 \
    --schedule 2 \
    --use_positional_embedding \
    --pos_num 25 \
    --fp16 \
    --use_side_transformers \
    --reduction_factor ${r} \
    --clip_model_name RN50x4 \
    --compute_time \
    --compute_memory \
    --seed ${seed} \
    ${@:5}  | tee $output/test_log.log

