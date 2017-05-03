#!/usr/bin/env bash

SNAPSHOT_DIR='snapshots/'
CHECKPOINT_DIR='models/model.ckpt-init'
PREDICTION_DIR='images_val/'
LOG_FILE=snapshots/test.log

# round 1
# OPTIMIZER="--optimizer=Adam --learning_rate=1e-3 --lr_decay_step=5000 --num_steps=30000 --momentum=0.9 --weight_decay=0.0005"
# mkdir snapshots
# python2 ./train_net.py $OPTIMIZER --snapshot_dir=$SNAPSHOT_DIR --restore_from=$CHECKPOINT_DIR 
python2 ./eval_net.py --save_dir=$PREDICTION_DIR --restore_from=$SNAPSHOT_DIR
# python2 ./calculate_mIU.py --pred="${PREDICTION_DIR}pred/" --gt="${PREDICTION_DIR}mask/" | tee $LOG_FILE
# echo $OPTIMIZER >> $LOG_FILE
# mv snapshots snapshots2

# round 2
# OPTIMIZER="--optimizer=SGD --learning_rate=1e-3 --lr_decay_step=6000 --num_steps=30000 --momentum=0.9 --weight_decay=0.0005"
# mkdir snapshots
# python2 ./train_net.py $OPTIMIZER --snapshot_dir=$SNAPSHOT_DIR --restore_from=$CHECKPOINT_DIR 
# python2 ./eval_net.py --save_dir=$PREDICTION_DIR --restore_from=$SNAPSHOT_DIR
# python2 ./calculate_mIU.py --pred="${PREDICTION_DIR}pred/" --gt="${PREDICTION_DIR}mask/" | tee $LOG_FILE
# echo $OPTIMIZER >> $LOG_FILE
# mv snapshots snapshots3
