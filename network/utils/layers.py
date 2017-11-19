# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf


def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    :param: x: Tensor, 4D BHWD input map
    :param: phase_train: boolean tf.Variable
    :return: normed: batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[x.get_shape().as_list()[-1]]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.get_shape().as_list()[-1]]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def activation(x, activation_type='relu'):
    if activation_type == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation_type == 'softmax':
        return tf.nn.softmax(x)
    elif activation_type == 'tanh':
        return tf.nn.tanh(x)
    else:
        return tf.nn.relu(x)


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, keep_prob_, padding='VALID', batch_normalization=True, phase=False):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    if batch_normalization:
        if phase is False:
            phase = tf.Variable(False, name='phase_train')
        conv_2d = batch_norm(conv_2d, phase)
        conv_2d = tf.contrib.layers.batch_norm(conv_2d, center=True, scale=True)
    return tf.nn.dropout(conv_2d, keep_prob_)


def deconv2d(x, W,stride, padding='VALID'):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, n, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding=padding)


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   


def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)


def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#   return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
