nice -n 10 python train.py \
--structure_string=64-64-64-64 \
--data_split_string_train=S1-S0-S5-S6-S7-S8 \
--data_split_string_test=S1 \
--batch_size=4 \
--joint_prob_max=3 \
--sigma=0.5 \
--gpu_string=0 \
--learning_rate=4e-6 

