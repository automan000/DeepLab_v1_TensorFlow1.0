#!/usr/bin/env bash
python ./train_net.py --snapshot_dir='snapshots/' --restore_from='models/model.ckpt-init'  --learning_rate=1e-3 --lr_decay_step=6000 --num_steps=30000 --momentum=0.9 --weight_decay=0.0005
python ./eval_net.py --save_dir='images_val/' --restore_from='snapshots/'
python ./calculate_mIU.py --pred='images_val/pred/' --gt='images_val/mask/' | tee test.log


