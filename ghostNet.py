from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dropout
from keras.utils.vis_utils import plot_model

from keras.layers import *
from keras.models import Model
from ghost_module import GhostModule
from keras import backend as K

class GhostNet(GhostModule):
    def __init__(self, shape):
        """Init"""
        super(GhostNet, self).__init__(shape)
        self.ratio = 2
        self.dw_kernel = 3

    def build(self):
        """创建GhostNet网络"""
        inputs = Input(shape=self.shape)
        x = Conv2D(16, 3, activation='relu')(inputs)
        x = Conv2D(32, 3, activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = self._ghost_bottleneck(x, 16, (3, 3), self.dw_kernel, 16, 1, self.ratio, True, name='ghost_bottleneck1')  #s=1的一个块，使用se
        x = Conv2D(16, 3, activation='relu')(x)
        x = Conv2D(32, 3, activation='relu')(x)
        x = Flatten()(x)
        activation = 'tanh'
        x = Dense(64, activation=activation)(x)
        x = Dense(64, activation=activation)(x)
        x = Dense(32, activation=activation)(x)

        model = Model(inputs, x)
        plot_model(model, to_file='ghostNet.png', show_shapes=True)
        return model
    
    
