nice -n 10 python train_2d.py \
--structure_string=64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--data_split_string_train_2d=2D \
--batch_size=2 \
--joint_prob_max=500 \
--sigma=1.5 \
--gpu_string=0-0 \
--learning_rate=5e-4 \
--train_2d=true \
--dataset_dir=./Dataset/ \
--load_ckpt_path=./tensor_record//tmp/model64-64-64.ckpt

