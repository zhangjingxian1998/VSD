output=weights/end2end_t5_b16_vsdv2
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
        --output $output ${@:2} \
        --load VLModel/snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 16 \
        --valid_batch_size 100 \
        --data VSDv2
        # --use_gold_rels
        # --test_only