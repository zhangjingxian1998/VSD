#/bin/bash
bash train_batch_16/baseline_bart.sh $1
wait
bash train_batch_16/end2end_bart.sh $1
wait
bash train_batch_16/baseline_t5.sh $1
wait
bash train_batch_16/end2end_t5.sh $1