nice -n 10 python testing.py \
--structure_string=64-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=2 \
--joint_prob_max=3 \
--sigma=1 \
--gpu_string=0 \
--train_2d=False \
--learning_rate=2e-4 \
--load_ckpt_path=./tensor_record/tmp/new12/model64-64.ckpt \

