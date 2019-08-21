rm ../results/$2_metrics.csv

#Uncomment this lines to evaluate the OVSD baseline by D. N. Rotman (2018)
#You need to have downloaded the results file on my drive (those with the word "baseline") and put them in the folder "../results/$1/"
#python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id baseline             --eval_model_leave_one_out 0  0 1 --model_name "Baseline"

python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_lstm_Holly2_7             --eval_model_leave_one_out 63  0 1 --model_name "LSTM-Res50 (Holly2)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_Holly2_7            --eval_model_leave_one_out 41  0 1 --model_name "Res50-Res50 (Holly2)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_youtLarg_7          --eval_model_leave_one_out 144 0 1 --model_name "Res50-Res50 (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res101_youtLarg_7         --eval_model_leave_one_out 157 0 1 --model_name "Res50-Res101 (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_convSco_youtLarg_7  --eval_model_leave_one_out 240 0 1 --model_name "Res50-Res50 SC (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_siam0005_youtLarg_7 --eval_model_leave_one_out 151 0 1 --model_name "Res50-Res50 Siam. (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_d1_youtLarg_7       --eval_model_leave_one_out 202 0 1 --model_name "Res50-Res50 DT (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_advL1_slowDisc_youtLarg_7       --eval_model_leave_one_out 163 0 1 --model_name "Res50-Res50 Adv. Loss (Youtube-large)"
python processResults.py --exp_id $1 --dataset_test $2 $3 --model_id res50_res50_biconvScoInitAtt_youtLarg_7     --eval_model_leave_one_out 462 0 1 --model_name "Res50-Res50 Deep-SA (Youtube-large)"
