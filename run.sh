#!/usr/bin/env bash
python2 ./train_net.py --snapshot_dir='snapshots/' --restore_from='models/model.ckpt-init'  --learning_rate=1e-3 --lr_decay_step=6000 --num_steps=30000 --momentum=0.95 --weight_decay=0.0005
python2 ./eval_net.py --save_dir='images_val/' --restore_from='snapshots/'
python2 ./calculate_mIU.py --pred='images_val/pred/' --gt='images_val/mask/' | tee test.log


