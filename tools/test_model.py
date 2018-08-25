#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-25 下午3:55
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_model.py.py
# @IDE: PyCharm
"""
测试cppn脚本
"""
import argparse

import tensorflow as tf
import cv2
import numpy as np

from cppn_model import cppn_net
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dims', type=int,
                        help='The dimension of embedding vectors',
                        default=8)
    parser.add_argument('--hidden_units_nums', type=int,
                        help='The number of hidden units',
                        default=32)
    parser.add_argument('--is_color', type=bool,
                        help='Whther to generate color image',
                        default=False)
    parser.add_argument('--img_h', type=int,
                        help='The output image height',
                        default=1060)
    parser.add_argument('--img_w', type=int,
                        help='The output image width',
                        default=1080)

    return parser.parse_args()


def minmax_scale_image(src_image, is_color=True):
    """
    重新归一化图像到0-255
    :param src_image:
    :param is_color:
    :return:
    """
    ret_image = np.array(1 - src_image)
    img_h = src_image.shape[0]
    img_w = src_image.shape[1]

    if not is_color:
        ret_image = np.array(ret_image.reshape((img_h, img_w)) * 255.0, dtype=np.uint8)
    else:
        ret_image = np.array(ret_image.reshape((img_h, img_w, 3)) * 255.0, dtype=np.uint8)

    return ret_image


def test_model(embedding_dims, is_color, hidden_nums, img_h, img_w):
    """

    :param embedding_dims:
    :param is_color:
    :param hidden_nums:
    :param img_h:
    :param img_w:
    :return:
    """
    with tf.device('/gpu:1'):
        net = cppn_net.CppnNet(embedding_dims=embedding_dims, is_colorful=is_color, scale=10, hidden_nums=hidden_nums)
        embeddings = net.generate_embeddings()
        net_out = net.generate_image(embedding_vecs=embeddings, image_width=img_w, image_height=img_h)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=False)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        sess.run(tf.global_variables_initializer())

        generated_image = sess.run(net_out)
        generated_image = np.squeeze(generated_image, 0)

        generated_image = minmax_scale_image(generated_image, is_color=is_color)

        cv2.imshow('result', generated_image)
        cv2.waitKey(5000)

    sess.close()


if __name__ == '__main__':
    # init args
    args = init_args()

    test_model(embedding_dims=args.embedding_dims, hidden_nums=args.hidden_units_nums,
               img_h=args.img_h, img_w=args.img_w, is_color=args.is_color)
