from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Reshape, Multiply
from keras.layers import Lambda, Concatenate

import tensorflow as tf

from keras import backend as K
import numpy as np
import math

class GhostModule:
    def __init__(self, shape):
        """初始化
        """
        self.shape = shape


    def _conv_block(self, inputs, outputs, kernel, strides, padding='same',
                    use_relu=True, use_bias=False):
        x = Conv2D(outputs, kernel, padding=padding, strides=strides, use_bias=use_bias)(inputs)
        x = BatchNormalization(axis=-1)(x)
        if use_relu:
            x = Activation('relu')(x)

        return x


    def _squeeze(self, inputs, exp, ratio, data_format='channels_last'):
        input_channels = int(inputs.shape[-1])
        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1,1,input_channels))(x)
        x = Conv2D(math.ceil(exp/ratio), (1,1), strides=(1,1), padding='same',
                   data_format=data_format, use_bias=False)(x)
        x = Activation('relu')(x)
        x = Conv2D(exp, (1,1),strides=(1,1), padding='same',
                   data_format=data_format, use_bias=False)(x)
        x = Activation('hard_sigmoid')(x)
        x = Multiply()([inputs, x])    # inputs和x逐元素相乘

        return x


    def _ghost_module(self, inputs, exp, kernel, dw_kernel, ratio, s=1,
                      padding='SAME',use_bias=False,
                      activation=None):
        output_channels = math.ceil(exp * 1.0 / ratio)
        x = Conv2D(output_channels, kernel, strides=(s, s), padding=padding,
                   activation=activation,
                   use_bias=use_bias)(inputs)
        if ratio == 1:
            return x
        dw = DepthwiseConv2D(dw_kernel, s, padding=padding, depth_multiplier=ratio-1,
                             activation=activation,
                             use_bias=use_bias)(x)
        x = Concatenate(axis=-1 )([x,dw])

        return x


    def _ghost_bottleneck(self, inputs, outputs, kernel, dw_kernel,
                          exp, s, ratio, squeeze, name=None):
        input_shape = K.int_shape(inputs)       # 获取输入张量的尺寸

        # 步长为1 且 输入通道数=输出通道数
        # print("步长为1 且 输入通道数=输出通道数:",s == 1 and input_shape[channel_axis] == outputs)
        if s == 1 and input_shape[-1] == outputs:
            res = inputs
        else:
            res = DepthwiseConv2D(kernel, strides=s, padding='SAME', depth_multiplier=ratio-1,
                                   activation=None, use_bias=False)(inputs)
            res = BatchNormalization(axis=-1)(res)
            res = self._conv_block(res, outputs, (1, 1), (1, 1), padding='valid',
                                   use_relu=False, use_bias=False)

        x = self._ghost_module(inputs, exp, [1,1], dw_kernel, ratio)

        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        if s > 1:
            x = DepthwiseConv2D(dw_kernel, s, padding='same', depth_multiplier=ratio-1,
                             activation=None, use_bias=False)(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
        if squeeze:
            x = self._squeeze(x, exp, 4)

        x = self._ghost_module(x, outputs, [1,1], dw_kernel, ratio)
        x = BatchNormalization(axis=-1)(x)


        x = Add()([res, x])

        return x







