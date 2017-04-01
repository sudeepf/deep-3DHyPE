nice -n 10 python train_2d.py \
--structure_string=64-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=2 \
--joint_prob_max=3 \
--sigma=0.5 \
--gpu_string=0 \
--learning_rate=4e-6 \
--train_2d=True \
--dataset_dir=./Dataset_2d/

