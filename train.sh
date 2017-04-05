nice -n 10 python train_2d.py \
--structure_string=64-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=4 \
--joint_prob_max=3 \
--sigma=1.2 \
--gpu_string=0-1 \
--learning_rate=2e-4 \
--train_2d=true \
--dataset_dir=./Dataset_2d/

