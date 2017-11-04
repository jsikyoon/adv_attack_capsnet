# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# This file is adapted from tensorflow official tutorial of mnist.
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import tensorflow as tf
from config import cfg
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data

from CapsNet import CapsNet

FLAGS = None


def model_test():
    model = CapsNet(None)
    model.creat_architecture()
    print("pass")


def main(_):
    # Max Epsilon
    eps = 2.0 * FLAGS.max_epsilon / 256.0 /FLAGS.max_iter;
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    tf.reset_default_graph()

    # Create the model
    caps_net = CapsNet(mnist)
    caps_net.creat_architecture()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_dir = cfg.TRAIN_DIR
    ckpt = tf.train.get_checkpoint_state(train_dir)

    # loss for the least label
    target = tf.one_hot(tf.argmin(caps_net._digit_caps_norm,1), 10, on_value=1.,off_value=0.)
    pos_loss = tf.maximum(0., cfg.M_POS - tf.reduce_sum(caps_net._digit_caps_norm * target,axis=1), name='pos_max')
    pos_loss = tf.square(pos_loss, name='pos_square')
    pos_loss = tf.reduce_mean(pos_loss)
    y_negs = 1. - target
    neg_loss = tf.maximum(0., caps_net._digit_caps_norm * y_negs - cfg.M_NEG)
    neg_loss = tf.reduce_sum(tf.square(neg_loss), axis=-1) * cfg.LAMBDA
    neg_loss = tf.reduce_mean(neg_loss)
    loss = pos_loss + neg_loss + cfg.RECONSTRUCT_W * caps_net.reconstruct_loss
    # step l.l and iter. l.l
    dy_dx,=tf.gradients(loss,caps_net._x);
    x_adv = tf.stop_gradient(caps_net._x -1*eps*tf.sign(dy_dx));
    x_adv = tf.clip_by_value(x_adv, 0., 1.);
   
    with tf.Session(config=config) as sess:
        if ckpt and cfg.USE_CKPT:
            print("Reading parameters from %s" % ckpt.model_checkpoint_path)
            caps_net.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created model with fresh paramters.')
            sess.run(tf.global_variables_initializer())
            print('Num params: %d' % sum(v.get_shape().num_elements()
                                         for v in tf.trainable_variables()))

        caps_net.train_writer.add_graph(sess.graph)
   
        #caps_net.adv_validation(sess, 'train',x_adv,FLAGS.max_iter)
        #caps_net.adv_validation(sess, 'validation',x_adv,FLAGS.max_iter)
        caps_net.adv_validation(sess, 'test',x_adv,FLAGS.max_iter,"samples/llcm_"+str(FLAGS.max_iter)+"_"+str(FLAGS.max_epsilon)+".PNG")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_DIR,
                        help='Directory for storing input data')
    parser.add_argument('--max_epsilon', type=int, default=10,
                        help='max_epsilon')
    parser.add_argument('--max_iter', type=int, default=1,
                        help='max iteration')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    # for model building test
    # model_test()
