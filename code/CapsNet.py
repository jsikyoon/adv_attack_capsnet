# ------------------------------------------------------------------
# Capsules_mnist
# By InnerPeace Wu
# ------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.contrib import slim
from tqdm import tqdm

from config import cfg

#Added by Jaesik Yoon
from get_data import image_save

def squash(cap_input):
    """
    squash function for keep the length of capsules between 0 - 1
    :arg
        cap_input: total input of capsules,
                   with shape: [None, h, w, c] or [None, n, d]
    :return
        cap_output: output of each capsules, which has the shape as cap_input
    """

    with tf.name_scope('squash'):
        # compute norm square of inputs with the last axis, keep dims for broadcasting
        # ||s_j||^2 in paper
        input_norm_square = tf.reduce_sum(tf.square(cap_input), axis=-1, keep_dims=True)

        # ||s_j||^2 / (1. + ||s_j||^2) * (s_j / ||s_j||)
        scale = input_norm_square / (1. + input_norm_square) / tf.sqrt(input_norm_square)

    return cap_input * scale


class CapsNet(object):
    def __init__(self, mnist):
        """initial class with mnist dataset"""
        self._mnist = mnist

        # keep tracking of the dimension of feature maps
        self._dim = 28
        # store number of capsules of each capsule layer
        # the conv1-layer has 0 capsules
        self._num_caps = [0]

    def _capsule(self, input, i_c, o_c, idx):
        """
        compute a capsule,
        conv op with kernel: 9x9, stride: 2,
        padding: VALID, output channels: 8 per capsule.
        As described in the paper.
        :arg
            input: input for computing capsule, shape: [None, w, h, c]
            i_c: input channels
            o_c: output channels
            idx: index of the capsule about to create

        :return
            capsule: computed capsule
        """
        with tf.variable_scope('cap_' + str(idx)):
            w = tf.get_variable('w', shape=[9, 9, i_c, o_c], dtype=tf.float32)
            cap = tf.nn.conv2d(input, w, [1, 2, 2, 1],
                               padding='VALID', name='cap_conv')
            if cfg.USE_BIAS:
                b = tf.get_variable('b', shape=[o_c, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                cap = cap + b
            # cap with shape [None, 6, 6, 8] for mnist dataset

            # Note: use "squash" as its non-linearity.
            capsule = squash(cap)
            # capsule with shape: [None, 6, 6, 8]
            # expand the dimensions to [None, 1, 6, 6, 8] for following concat
            capsule = tf.expand_dims(capsule, axis=1)

            # return capsule with shape [None, 1, 6, 6, 8]
            return capsule

    def _dynamic_routing(self, primary_caps, layer_index):
        """"
        dynamic routing between capsules
        :arg
            primary_caps: primary capsules with shape [None, 1, 32 x 6 x 6, 1, 8]
            layer_index: index of the current capsule layer, i.e. the input layer for routing
        :return
            digit_caps: the output of digit capsule layer output, with shape: [None, 10, 16]
        """
        # number of the capsules in current layer
        num_caps = self._num_caps[layer_index]
        # weight matrix for capsules in "layer_index" layer
        # W_ij
        cap_ws = tf.get_variable('cap_w', shape=[10, num_caps, 8, 16],
                                 dtype=tf.float32,
                                 )
        # initial value for "tf.scan", see official doc for details
        fn_init = tf.zeros([10, num_caps, 1, 16])

        # x after tiled with shape: [10, num_caps, 1, 8]
        # cap_ws with shape: [10, num_caps, 8, 16],
        # [8 x 16] for each pair of capsules between two layers
        # u_hat_j|i = W_ij * u_i
        cap_predicts = tf.scan(lambda ac, x: tf.matmul(x, cap_ws),
                               tf.tile(primary_caps, [1, 10, 1, 1, 1]),
                               initializer=fn_init, name='cap_predicts')
        # cap_predicts with shape: [None, 10, num_caps, 1, 16]
        cap_predictions = tf.squeeze(cap_predicts, axis=[3])
        # after squeeze with shape: [None, 10, num_caps, 16]

        # log prior probabilities
        log_prior = tf.get_variable('log_prior', shape=[10, num_caps], dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=cfg.PRIOR_TRAINING)
        # log_prior with shape: [10, num_caps]
        # V1. static way
        if cfg.ROUTING_WAY == 'static':
            digit_caps = self._dynamic_routingV1(log_prior, cap_predictions)
        # V2. dynamic way
        elif cfg.ROUTING_WAY == 'dynamic':
            digit_caps = self._dynamic_routingV2(log_prior, cap_predictions, num_caps)
        else:
            raise NotImplementedError

        return digit_caps

    def _dynamic_routingV2(self, prior, cap_predictions, num_caps):
        """
        doing dynamic routing with tf.while_loop
        :arg
            proir: log prior for scaling with shape [10, num_caps]
            cap_prediction: predictions from layer below with shape [None, 10, num_caps, 16]
            num_caps: num_caps
        :return
            digit_caps: digit capsules with shape [None, 10, 16]
        """
        # check V1 for implementation details
        init_cap = tf.reduce_sum(cap_predictions, -2)
        iters = tf.constant(cfg.ROUTING_ITERS)
        prior = tf.expand_dims(prior, 0)

        def body(i, prior, cap_out):
            c = tf.nn.softmax(prior, dim=1)
            c_expand = tf.expand_dims(c, axis=-1)
            s_t = tf.multiply(cap_predictions, c_expand)
            s = tf.reduce_sum(s_t, axis=[2])
            cap_out = squash(s)
            delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(cap_out, axis=2),
                                                    cap_predictions),
                                        axis=[-1])
            prior = prior + delta_prior

            return [i - 1, prior, cap_out]

        condition = lambda i, proir, cap_out: i > 0
        _, prior, digit_caps = tf.while_loop(condition, body, [iters, prior, init_cap],
                                             shape_invariants=[iters.get_shape(),
                                                               tf.TensorShape([None, 10, num_caps]),
                                                               init_cap.get_shape()])

        return digit_caps

    def _dynamic_routingV1(self, prior, cap_predictions):
        """
        doing dynamic routing with for loop as static implementation
        :arg
            proir: log prior for scaling with shape [10, num_caps]
            cap_prediction: predictions from layer below with shape [None, 10, num_caps, 16]
        :return
            digit_caps: digit capsules with shape [None, 10, 16]
        """
        prior = tf.expand_dims(prior, 0)
        # prior shape: [1, 10, num_caps]
        for idx in xrange(cfg.ROUTING_ITERS):
            with tf.name_scope('routing_%s' % idx):

                c = tf.nn.softmax(prior, dim=1)
                # c shape: [1, 10, num_caps]
                c_t = tf.expand_dims(c, axis=-1)
                # c_t shape: [1, 10, num_caps, 1]

                s_t = tf.multiply(cap_predictions, c_t)
                # s_t shape: [None, 10, num_caps, 16]
                # for each capsule in the layer after, add all the weighted capsules to get
                # the capsule input for it.
                # s_j = Sum_i (c_ij u_hat_j|i)
                s = tf.reduce_sum(s_t, axis=[2])

                # s shape: [None, 10, 16]
                digit_caps = squash(s)
                # digit_caps shape: [None, 10, 16]

                # u_hat_j|i * v_j
                delta_prior = tf.reduce_sum(tf.multiply(tf.expand_dims(digit_caps, axis=2),
                                                        cap_predictions),
                                            axis=[-1])
                # delta_prior shape: [None, 10, num_caps]

                prior = prior + delta_prior

        # shape [None, 10, 16]
        return digit_caps

    def _reconstruct(self, digit_caps):
        """
        reconstruct from digit capsules with 3 fully connected layer
        :param
            digit_caps: digit capsules with shape [None, 10, 16]
        :return:
            out: out of reconstruction
        """
        # (TODO wu) there is two ways to do reconstruction.
        # 1. only use the target capsule with dimension [None, 16] or [16,] (use it for default)
        # 2. use all the capsule, including the masked out ones with lots of zeros
        with tf.name_scope('reconstruct'):
            y_ = tf.expand_dims(self._y_, axis=2)
            # y_ shape: [None, 10, 1]

            # for method 1.
            target_cap = y_ * digit_caps
            # target_cap shape: [None, 10, 16]
            target_cap = tf.reduce_sum(target_cap, axis=1)
            # target_cap: [None, 16]

            # for method 2.
            # target_cap = tf.reshape(y_ * digit_caps, [-1, 10*16])

            fc = slim.fully_connected(target_cap, 512,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 1024,
                                      weights_initializer=self._w_initializer)
            fc = slim.fully_connected(fc, 784,
                                      weights_initializer=self._w_initializer,
                                      activation_fn=None)
            # the last layer with sigmoid activation
            out = tf.sigmoid(fc)
            # out with shape [None, 784]

            return out

    def _add_loss(self, digit_caps):
        """
        add the margin loss and reconstruction loss
        :arg
            digit_caps: output of digit capsule layer, shape [None, 10, 16]
        :return
            total_loss:
        """
        with tf.name_scope('loss'):
            # [None, 10 , 1]
            # y_ = tf.expand_dims(self._y_, axis=2)
            self._digit_caps_norm = tf.norm(digit_caps, ord=2, axis=2,
                                            name='digit_caps_norm')
            # digit_caps_norm shape: [None, 10]
            # loss of positive classes
            # max(0, m+ - ||v_c||) ^ 2
            with tf.name_scope('pos_loss'):
                pos_loss = tf.maximum(0., cfg.M_POS - tf.reduce_sum(self._digit_caps_norm * self._y_,
                                                                    axis=1), name='pos_max')
                pos_loss = tf.square(pos_loss, name='pos_square')
                pos_loss = tf.reduce_mean(pos_loss)
            tf.summary.scalar('pos_loss', pos_loss)
            # pos_loss shape: [None, ]

            # get index of negative classes
            y_negs = 1. - self._y_
            # max(0, ||v_c|| - m-) ^ 2
            with tf.name_scope('neg_loss'):
                neg_loss = tf.maximum(0., self._digit_caps_norm * y_negs - cfg.M_NEG)
                neg_loss = tf.reduce_sum(tf.square(neg_loss), axis=-1) * cfg.LAMBDA
                neg_loss = tf.reduce_mean(neg_loss)
            tf.summary.scalar('neg_loss', neg_loss)
            # neg_loss shape: [None, ]

            reconstruct = self._reconstruct(digit_caps)

            # loss of reconstruction
            with tf.name_scope('l2_loss'):
                reconstruct_loss = tf.reduce_sum(tf.square(self._x - reconstruct), axis=-1)
                reconstruct_loss = tf.reduce_mean(reconstruct_loss)
            tf.summary.scalar('reconstruct_loss', reconstruct_loss)

            total_loss = pos_loss + neg_loss + \
                         cfg.RECONSTRUCT_W * reconstruct_loss
            #Added by Jaesik Yoon
            self.reconstruct_loss=reconstruct_loss;
            tf.summary.scalar('loss', total_loss)

        return total_loss

    def creat_architecture(self):
        """creat architecture of the whole network"""
        # set up placeholder of input data and labels
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._y_ = tf.placeholder(tf.float32, [None, 10])

        # set up initializer for weights and bias
        self._w_initializer = tf.truncated_normal_initializer(stddev=0.1)
        self._b_initializer = tf.zeros_initializer()

        with tf.variable_scope('CapsNet', initializer=self._w_initializer):
            # build net
            self._build_net()

            # set up exponentially decay learning rate
            self._global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(cfg.LR, self._global_step,
                                                       cfg.STEP_SIZE, cfg.DECAY_RATIO,
                                                       staircase=True)
            tf.summary.scalar('learning rate', learning_rate)

            # set up adam optimizer with default setting
            self._optimizer = tf.train.AdamOptimizer(learning_rate)
            gradidents = self._optimizer.compute_gradients(self._loss)
            tf.summary.scalar('grad_norm', tf.global_norm(gradidents))

            self._train_op = self._optimizer.apply_gradients(gradidents,
                                                             global_step=self._global_step)
            # set up accuracy ops
            self._accuracy()
            self._summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            # set up summary writer
            self.train_writer = tf.summary.FileWriter(cfg.TB_DIR + '/train')
            self.val_writer = tf.summary.FileWriter(cfg.TB_DIR + '/val')

    def _build_net(self):
        """build the graph of the network"""

        # reshape for conv ops
        with tf.name_scope('x_reshape'):
            x_image = tf.reshape(self._x, [-1, 28, 28, 1])

        # initial conv1 op
        # 1). conv1 with kernel 9x9, stride 1, output channels 256
        with tf.variable_scope('conv1'):
            # specially initialize it with xavier initializer with no good reason.
            w = tf.get_variable('w', shape=[9, 9, 1, 256], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            # conv op
            conv1 = tf.nn.conv2d(x_image, w, [1, 1, 1, 1],
                                 padding='VALID', name='conv1')
            if cfg.USE_BIAS:
                # bias (TODO wu) no idea if the paper uses bias or not
                b = tf.get_variable('b', shape=[256, ], dtype=tf.float32,
                                    initializer=self._b_initializer)
                conv1 = tf.nn.relu(conv1 + b)
            else:
                conv1 = tf.nn.relu(conv1)

            # update dimensions of feature map
            self._dim = (self._dim - 9) // 1 + 1
            assert self._dim == 20, "after conv1, dimensions of feature map" \
                                    "should be 20x20"

            # conv1 with shape [None, 20, 20, 256]

        # build up primary capsules
        with tf.variable_scope('PrimaryCaps'):

            # update dim of capsule grid
            self._dim = (self._dim - 9) // 2 + 1
            # number of primary caps: 6x6x32 = 1152
            self._num_caps.append(self._dim ** 2 * cfg.PRIMARY_CAPS_CHANNELS)
            assert self._dim == 6, "dims for primary caps grid should be 6x6."

            # build up PriamryCaps with 32 channels and 8-D vector
            # 1. dummy solution
            '''
            caps = []
            for idx in xrange(cfg.PRIMARY_CAPS_CHANNELS):
                # get a capsule with 8-D
                cap = self._capsule(conv1, 256, 8, idx)
                # cap with shape: [None, 1, 6, 6, 8]
                caps.append(cap)

            # concat all the primary capsules
            primary_caps = tf.concat(caps, axis=1)
            # primary_caps with shape: [None, 32, 6, 6, 8]
            with tf.name_scope('primary_cap_reshape'):
                # reshape and expand dims for broadcasting in dynamic routing
                primary_caps = tf.reshape(primary_caps,
                                                  shape=[-1, 1, self._num_caps[1], 1, 8])
                # primary_caps with shape: [None, 1, 1152, 1, 8]
            '''
            # 2. faster one
            primary_caps = slim.conv2d(conv1, 32*8, 9, 2, padding='VALID', activation_fn=None)
            primary_caps = tf.reshape(primary_caps, [-1, 1, self._num_caps[1], 1, 8])
            primary_caps = squash(primary_caps)

        # dynamic routing
        with tf.variable_scope("digit_caps"):
            self._digit_caps = self._dynamic_routing(primary_caps, 1)

        # set up losses
        self._loss = self._add_loss(self._digit_caps)

    def _accuracy(self):
        with tf.name_scope('accuracy'):
            # digit_caps_norm = tf.norm(self._digit_caps, ord=2, axis=-1)
            correct_prediction = tf.equal(tf.argmax(self._y_, 1),
                                          tf.argmax(self._digit_caps_norm, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)
            tf.summary.scalar('accuracy', self.accuracy)

    def train_with_summary(self, sess, batch_size=100, iters=0):
        batch = self._mnist.train.next_batch(batch_size)
        loss, _, train_acc, train_summary = sess.run([self._loss, self._train_op,
                                                      self.accuracy, self._summary_op],
                                                     feed_dict={self._x: batch[0],
                                                                self._y_: batch[1]})
        if iters % cfg.PRINT_EVERY == 0 and iters > 0:
            val_batch = self._mnist.validation.next_batch(batch_size)

            self.train_writer.add_summary(train_summary, iters)
            self.train_writer.flush()

            print("iters: %d / %d, loss ==> %.4f " % (iters, cfg.MAX_ITERS, loss))
            print('train accuracy: %.4f' % train_acc)

            test_acc, test_summary = sess.run([self.accuracy, self._summary_op],
                                              feed_dict={self._x: val_batch[0],
                                                         self._y_: val_batch[1]})
            print('val   accuracy: %.4f' % test_acc)
            self.val_writer.add_summary(test_summary, iters)
            self.val_writer.flush()

        if iters % cfg.SAVE_EVERY == 0 and iters > 0:
            self.snapshot(sess, iters=iters)
            self.test(sess)

    def snapshot(self, sess, iters=0):
        save_path = cfg.TRAIN_DIR + '/capsnet'
        self.saver.save(sess, save_path, iters)

    def test(self, sess, set='validation'):
        if set == 'test':
            x = self._mnist.test.images
            y_ = self._mnist.test.labels
        else:
            x = self._mnist.validation.images
            y_ = self._mnist.validation.labels
       
        acc = []
        for i in tqdm(xrange(len(x) // 100), desc="calculating %s accuracy" % set):
            x_i = x[i * 100: (i + 1) * 100]
            y_i = y_[i * 100: (i + 1) * 100]
            ac = sess.run(self.accuracy,
                          feed_dict={self._x: x_i,
                                     self._y_: y_i})
            acc.append(ac)
        all_ac = np.mean(np.array(acc))
        print("whole {} accuracy: {}".format(set, all_ac))
   
    # added by Jaesik Yoon 
    def adv_validation(self, sess, set, x_adv, max_iter,fname=None):
        if set == 'test':
            x = self._mnist.test.images
            y_ = self._mnist.test.labels
        if set == 'validation':
            x = self._mnist.validation.images
            y_ = self._mnist.validation.labels
        if set == 'train':
            x = self._mnist.validation.images
            y_ = self._mnist.validation.labels
        
        acc = []
        for i in tqdm(xrange(len(x) // 100), desc="calculating %s accuracy" % set):
            x_i = x[i * 100: (i + 1) * 100]
            y_i = y_[i * 100: (i + 1) * 100]
            for j in range(max_iter):
              x_i=sess.run(x_adv,feed_dict={self._x:x_i,self._y_:y_i});
            
            # When writing imgs
            image_save(x_i,fname);
            return;
            
            ac = sess.run(self.accuracy,
                          feed_dict={self._x: x_i,
                                     self._y_: y_i})
            acc.append(ac)
        all_ac = np.mean(np.array(acc))
        print("whole {} accuracy: {}".format(set, all_ac))
