output=weights/end2end_t5
PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    VLModel/src/vrd_caption.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 20 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load weights/end2end_t5_b16/BEST \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --test_only \
        $2
        