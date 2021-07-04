"""
Part of this code is based on the Facebook fastMRI code.

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
import tensorflow as tf
from torch import nn
from torch.nn import functional as F


def policy(ne_img, args):

    conv1 = tf.layers.conv2d(
        inputs=ne_img,
        filters=args.filters,
        kernel_size=3,
        strides=1,
        padding='valid',  # 边缘不填充
        name='conv1')

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=args.filters,
        kernel_size=3,
        strides=1,
        padding='valid',
        name='conv2')

    flat = tf.layers.flatten(conv2)

    dense = tf.layers.dense(
        inputs=flat,
        units=10,
        activation=None,
        name='fc')

    logits = tf.layers.dense(
        inputs=dense,
        units=1,
        name='logits')

    value = tf.layers.dense(
        inputs=dense,
        units=1,
        name='value')

    aprob = tf.nn.softmax(logits)
    action_logprob = tf.nn.log_softmax(logits)

    pol = tf.distributions.Categorical(aprob)

    return pol


class BasicBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection
    before the last ReLU activation.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out
