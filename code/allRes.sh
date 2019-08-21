evalOneDataset () {

  #First arg in the name of the experience (e.g. for an evaluation on the BBC dataset you can call it "evalBBC")
  #Second arg is the name of the dataset (e.g. "bbc" if you want to run an eval on the BBC dataset)

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 63  --start_mode fine_tune --init_path ../models/keepLearning3/modellstm_res50_holly2_epoch63   --model_id res50_lstm_Holly2_7

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 41  --start_mode fine_tune --init_path ../models/keepLearning3/modelres50_res50_holly2_epoch41  --model_id res50_res50_Holly2_7    --temp_model resnet50

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 144 --start_mode fine_tune --init_path ../models/youtLarg/modelres50_res50_yout_epoch144        --model_id res50_res50_youtLarg_7  --temp_model resnet50

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 157 --start_mode fine_tune --init_path ../models/youtLarg/modelres101_res50_yout_epoch157       --model_id res50_res101_youtLarg_7 --temp_model resnet50 --feat resnet101

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 240 --start_mode fine_tune --init_path ../models/youtLarg/modelresnet50_res50_SC_yout_epoch240  --model_id res50_res50_convSco_youtLarg_7  --temp_model resnet50 --score_conv_wind_size 9

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 151 --start_mode fine_tune --init_path ../models/youtLarg/modelres50_res50_Siam_yout_epoch151   --model_id res50_res50_siam0005_youtLarg_7  --temp_model resnet50

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 202 --start_mode fine_tune --init_path ../models/youtLarg/modelres50_res50_DT_yout_epoch202     --model_id res50_res50_d1_youtLarg_7  --temp_model resnet50

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 163 --start_mode fine_tune --init_path ../models/youtLarg/modelres50_res50_adv_yout_epoch163    --model_id res50_res50_advL1_slowDisc_youtLarg_7  --temp_model resnet50

  python trainVal.py -c model.config --exp_id $1  --train_part_end 0 --dataset_val $2 --val_part_beg 0 \
                                     --epochs 462 --start_mode fine_tune --init_path ../models/youtLarg/modelresnet50_biconvScoInitAtt_epoch462    --model_id res50_res50_biconvScoInitAtt_youtLarg_7  --temp_model resnet50 \
  				                           --score_conv_wind_size 7 \
                                     --score_conv_bilay True \
                                     --score_conv_chan 8 \
                                     --score_conv_attention True
}

evalOneDataset $1 $2
