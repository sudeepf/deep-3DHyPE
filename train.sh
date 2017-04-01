nice -n 10 python train_2d.py \
--structure_string=64-64 \
--data_split_string_train=S1-S0-S5-S6-S7-S8 \
--data_split_string_test=S1 \
--batch_size=4 \
--joint_prob_max=1 \
--sigma=0.75 \
--gpu_string=0-1 \
--learning_rate=5e-5 \
--train_2d=False \
--dataset_dir=./Dataset/

