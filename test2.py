"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
import os
import time
import math
import scipy.misc
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import scipy.misc

from nets.large_fov.image_reader_for_test import ImageReader
from nets.large_fov.model import DeepLabLFOVModel
from nets.large_fov.utils import decode_labels

BATCH_SIZE = 10
DATA_DIRECTORY = '/home/automan/Data/Pascal/VOC2012'
DATA_LIST_PATH = './dataset/val.txt'
INPUT_SIZE = '321,321'

MEAN_IMG = tf.Variable(np.array((104.00698793,116.66876762,122.67891434)), trainable=False, dtype=tf.float32)
RANDOM_SCALE = False
RESTORE_FROM = './snapshots/checkpoint'
SAVE_DIR = './images_val/'
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
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
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

    h, w = map(int, args.input_size.split(','))
    input_size = None  # (h, w)
    
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
        image_batch, label_batch, shape_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabLFOVModel(args.weights_path)

    pred = net.preds(image_batch)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    ckpt = tf.train.get_checkpoint_state('./snapshots/')
    load(saver, sess, ckpt.model_checkpoint_path)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir + 'mask/'):
        os.makedirs(args.save_dir + 'mask/')
    if not os.path.exists(args.save_dir + 'pred/'):
        os.makedirs(args.save_dir + 'pred/')

    # Iterate over training steps.
    for step in range(int(math.ceil(reader.image_num / args.batch_size))):
        start_time = time.time()
        images, labels, shapes, preds = sess.run([image_batch, label_batch, shape_batch, pred])
        for i in range(len(preds)):
            shape = shapes[i]
            label = (labels[i])[:shape[0], :shape[1], :]
            prediction = (preds[i])[:shape[0], :shape[1], :]
            # vlabel = decode_labels(labels[i, :, :, 0])[:shape[0], :shape[1], :]
            # vprediction = decode_labels(preds[i, :, :, 0])[:shape[0],:shape[1],:]

            scipy.misc.imsave(args.save_dir + 'mask/' + str(step * args.batch_size + i) +'.png', label)
            scipy.misc.imsave(args.save_dir + 'pred/' + str(step * args.batch_size + i) +'.png', prediction)


        duration = time.time() - start_time
        print('step {:d} \t ({:.3f} sec/step)'.format(step, duration))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
