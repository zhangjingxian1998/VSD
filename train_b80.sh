#/bin/bash
bash train_batch_80/baseline_bart.sh $1
wait
bash train_batch_80/end2end_bart.sh $1
wait
bash train_batch_80/baseline_t5.sh $1
wait
bash train_batch_80/end2end_t5.sh $1