output=weights/baseline_bart
PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    VLModel/src/caption_sp.py \
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
        --backbone 'facebook/bart-base' \
        --output $output \
        --load weights/baseline_bart_b80/BEST \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --test_only \