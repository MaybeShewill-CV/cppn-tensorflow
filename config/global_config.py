#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-8-25 下午1:50
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : global_config.py
# @IDE: PyCharm
"""
设置一些模型超参数
"""
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 8
