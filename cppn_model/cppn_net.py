#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-25 下午1:55
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : cppn_net.py
# @IDE: PyCharm
"""
cppn模型
"""
import numpy as np
import tensorflow as tf

from cppn_model import cnn_basenet


class CppnNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, embedding_dims, is_colorful, scale, hidden_nums):
        """

        :param embedding_dims: embedding vector length
        :param is_colorful: whether generate colourful images
        :param scale: the scale of
        :param hidden_nums:
        """
        super(CppnNet, self).__init__()
        self._z_dims = embedding_dims
        if is_colorful:
            self._channels = 3
        else:
            self._channels = 1
        self._scale = scale
        self._hidden_nums = hidden_nums

    def _compute_coordinates(self, image_width, image_height, scale=1.0):
        """
        计算图像x, y坐标点和相对应的半径
        :param image_width: 原始图像宽
        :param image_height: 原始图像高
        :param scale:
        :return:
        """
        n_points = image_width * image_height
        x_range = scale * (np.arange(image_width) - (image_width - 1) / 2.0) / (image_width - 1) / 0.5
        y_range = scale * (np.arange(image_height) - (image_height - 1) / 2.0) / (image_height - 1) / 0.5
        x_vec = np.matmul(np.ones((image_height, 1)), x_range.reshape((1, image_width)))
        y_vec = np.matmul(y_range.reshape((image_height, 1)), np.ones((1, image_width)))
        r_vec = np.sqrt(x_vec * x_vec + y_vec * y_vec)
        x_vec = np.tile(x_vec.flatten(), 1).reshape(1, n_points, 1)
        y_vec = np.tile(x_vec.flatten(), 1).reshape(1, n_points, 1)

        r_vec = np.tile(r_vec.flatten(), 1).reshape(1, n_points, 1)

        return np.array(x_vec, np.float32),np.array(y_vec, np.float32), np.array(r_vec, np.float32)

    def _generator(self, embedding_vecs, x_vec, y_vec, r_vec,
                   image_width, image_height, reuse=False):
        """
        定义生成器
        :param embedding_vecs:
        :param x_vec:
        :param y_vec:
        :param r_vec:
        :param image_width:
        :param image_height:
        :param reuse:
        :return:
        """
        with tf.variable_scope('generator', reuse=reuse):
            n_points = image_width * image_height

            z_scaled = tf.reshape(embedding_vecs, [1, 1, self._z_dims]) * \
                       tf.ones([n_points, 1], dtype=tf.float32) * self._scale
            z_flatten = tf.reshape(z_scaled, [1 * n_points, self._z_dims])
            x_flatten = tf.reshape(x_vec, [1 * n_points, 1])
            y_flatten = tf.reshape(y_vec, [1 * n_points, 1])
            r_flatten = tf.reshape(r_vec, [1 * n_points, 1])

            z_feats = self.fullyconnect(inputdata=z_flatten, out_dim=self._hidden_nums,
                                        use_bias=False, name='g_z_feats')
            x_feats = self.fullyconnect(inputdata=x_flatten, out_dim=self._hidden_nums,
                                        use_bias=False, name='g_x_feats')
            y_feats = self.fullyconnect(inputdata=y_flatten, out_dim=self._hidden_nums,
                                        use_bias=False, name='g_y_feats')
            r_feats = self.fullyconnect(inputdata=r_flatten, out_dim=self._hidden_nums,
                                        use_bias=False, name='g_r_feats')
            merge_feats = z_feats + x_feats + y_feats + r_feats

            generate_feats = self._generate_func_v1(merge_feats)
            # generate_feats = self._generate_func_v2(merge_feats)
            # generate_feats = self._generate_func_v3(merge_feats)

            ret = tf.reshape(generate_feats, [1, image_height, image_width, self._channels])

            return ret

    def _generate_func_v1(self, merge_feats):
        """
        不同的生成模板
        :param merge_feats:
        :return:
        """
        feats = self.tanh(merge_feats, name='g_func_tanh_1')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_1')
        feats = self.softplus(feats, name='g_func_softplus_1')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_2')
        feats = self.tanh(feats, name='g_func_tanh_2')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_3')
        feats = self.softplus(feats, name='g_func_softplus_2')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_4')
        feats = self.tanh(feats, name='g_func_tanh_3')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_5')
        feats = self.softplus(feats, name='g_func_softplus_3')
        feats = self.fullyconnect(feats, self._channels, use_bias=False, name='g_func_fc_6')
        output = self.sigmoid(feats, name='g_func_output')

        return output

    def _generate_func_v2(self, merge_feats):
        """
        不同的生成模板
        :param merge_feats:
        :return:
        """
        feats = self.tanh(merge_feats, name='g_func_tanh_1')
        for i in range(3):
            feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False,
                                      name='g_func_fc_{:d}'.format(i + 1))
            feats = self.tanh(feats, name='g_func_tanh_{:d}'.format(i + 2))
        feats = self.fullyconnect(feats, self._channels, use_bias=False,
                                  name='g_func_fc_4')
        output = self.sigmoid(feats, name='g_func_output')

        return output

    def _generate_func_v3(self, merge_feats):
        """
        不同的生成模板
        :param merge_feats:
        :return:
        """
        feats = self.tanh(merge_feats, name='g_func_tanh_1')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_1')
        feats = self.softplus(feats, name='g_func_softplus_1')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_2')
        feats = self.tanh(feats, name='g_func_tanh_2')
        feats = self.fullyconnect(feats, self._hidden_nums, use_bias=False, name='g_func_fc_3')
        feats = self.softplus(feats, name='g_func_softplus_3')
        feats = self.fullyconnect(feats, self._channels, use_bias=False, name='g_func_fc_4')
        output = 0.5 * tf.sin(feats) + 0.5

        return output

    def generate_embeddings(self):
        """

        :return:
        """
        np.random.seed(1234)
        return np.random.uniform(-1.0, 1.0, size=(1, self._z_dims)).astype(np.float32)

    def generate_image(self, embedding_vecs, image_width, image_height):
        """
        生成接口
        :param embedding_vecs:
        :param image_width:
        :param image_height:
        :return:
        """
        if embedding_vecs is None:
            embedding_vecs = np.random.uniform(-1.0, 1.0, size=(1, self._z_dims)).astype(np.float32)

        x_vec, y_vec, r_vec = self._compute_coordinates(image_height=image_height,
                                                        image_width=image_width,
                                                        scale=self._scale)
        ret_image = self._generator(embedding_vecs=embedding_vecs, x_vec=x_vec, y_vec=y_vec, r_vec=r_vec,
                                    image_height=image_height, image_width=image_width, reuse=False)

        return ret_image
