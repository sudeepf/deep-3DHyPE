nice -n 10 python train_2d.py \
--structure_string=64-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=16 \
--joint_prob_max=1 \
--sigma=2. \
--gpu_string=0-1 \
--learning_rate=8e-4 \
--train_2d=true \
--dataset_dir=./Dataset_2d/ \
--load_ckpt_path=./tensor_record//tmp/model64-64.ckpt

