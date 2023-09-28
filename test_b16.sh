#/bin/bash
bash test_batch_16/baseline_bart.sh $1 >> test_16_res/baseline_bart.log 2>&1 &
wait
bash test_batch_16/end2end_bart.sh $1 >> test_16_res/end2end_bart.log 2>&1 &
wait
bash test_batch_16/baseline_t5.sh $1 >> test_16_res/baseline_t5.log 2>&1 &
wait
bash test_batch_16/end2end_t5.sh $1 >> test_16_res/end2end_t5.log 2>&1 &
wait
bash test_batch_16/end2end_bart.sh $1 $2 >> test_16_res/end2end_bart_golden.log 2>&1 &
wait
bash test_batch_16/end2end_t5.sh $1 $2 >> test_16_res/end2end_t5_golden.log 2>&1 &