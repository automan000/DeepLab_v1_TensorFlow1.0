"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from nets.large_fov.image_reader import ImageReader
from nets.large_fov.model import DeepLabLFOVModel
from nets.large_fov.utils import decode_labels

BATCH_SIZE = 10
DATA_DIRECTORY = '/home/automan/Data/Pascal/VOC2012'
DATA_LIST_PATH = './dataset/train.txt'
INPUT_SIZE = '321,321'
OPTIMIZER = 'SGD'
LEARNING_RATE = 1e-3
WEIGHT_DECAY_FACTOR = 0.0005
LR_DECAY_EVERY = 5000
MOMENTUM = 0.9
MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000
RANDOM_SCALE = True
RESTORE_FROM = './models/model.ckpt-init'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/'
LOG_DIR = './logs/'
WEIGHTS_PATH = None

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER,
                        help="Optimizer (SGD or Adam).")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--lr_decay_step", type=int, default=LR_DECAY_EVERY,
                        help="Learning rate decays every n steps.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_FACTOR,
                        help="Weights decay factor.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="momentum.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Where to save logs.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
    global_step = tf.Variable(0, trainable=False)

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            RANDOM_SCALE,
            coord)
        image_batch, label_batch, _ = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabLFOVModel(args.weights_path)
    # Define the loss and optimisation parameters.
    loss = net.loss(image_batch, label_batch)
    tf.summary.scalar('loss', loss)

    trainable = tf.trainable_variables()
    # Weight decay
    decays = tf.reduce_sum(
        input_tensor=args.weight_decay * tf.stack(
            [tf.nn.l2_loss(i) for i in trainable]
        ),
        name='weights_norm'
    )
    tf.summary.scalar('weight_decay', decays)
    loss += decays
    tf.summary.scalar('total_loss', loss)

    if args.optimizer == 'SGD':
        print('Apply SGD optimizer')
        lr = tf.train.exponential_decay(args.learning_rate, global_step, args.lr_decay_step, 0.1)
        tf.summary.scalar('learning_rate', lr)
        # train_op1 = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step, var_list=trainable[:-2])
        # train_op2 = tf.train.AdamOptimizer(lr * 10).minimize(loss, global_step=global_step, var_list=trainable[-2:])
        train_op1 = tf.train.MomentumOptimizer(lr, momentum=args.momentum).minimize(loss, global_step=global_step,
                                                                          var_list=trainable[:-2])
        # the lr for the final classifier layer is 10x greater than other layers.
        train_op2 = tf.train.MomentumOptimizer(lr * 10, momentum=args.momentum).minimize(loss, global_step=global_step,
                                                                               var_list=trainable[-2:])
        optimizer = tf.group(train_op1, train_op2)
        tf.summary.scalar('learning_rate', lr)
    elif args.optimizer == 'Adam':
        print('Apply Adam optimizer')
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, global_step=global_step,
                                                                          var_list=trainable[:-2])
        # tf.summary.scalar('learning_rate', optimizer._lr_t)
    else:
        sys.exit("Unknown optimizer.")


    pred = net.preds(image_batch)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)


    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=40)
    if args.restore_from is not None:
        load(saver, sess, args.restore_from)

    now_time = datetime.now()
    train_writer = tf.summary.FileWriter(args.log_dir + str(now_time), sess.graph)
    summary_op = tf.summary.merge_all()

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
   
    # Iterate over training steps.
    for step in range(args.num_steps +1):  # save the last checkpoint, do this trick with '+1'
        start_time = time.time()
        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, summary, _ = sess.run([loss, image_batch, label_batch, pred, summary_op, optimizer])
            train_writer.add_summary(summary, global_step=step)
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([loss, optimizer])
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
