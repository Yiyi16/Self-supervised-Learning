
from mxnet.gluon import nn


class ConvBlock(nn.HybridBlock):
    """
    Conv + bn + act.
    """
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 use_bias=True, activation=None, downsample=False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        # Network structure
        with self.name_scope():
            # 1. Convolution block
            self.__conv = nn.Conv2D(channels, kernel_size, strides=strides, padding=padding, use_bias=use_bias)
            self.__bn = nn.BatchNorm()
            self.__act = nn.Activation(activation) if activation is not None else None
            # 2. Down sample.
            self.__downsample = nn.MaxPool2D(pool_size=3, strides=2) if downsample else None

    def hybrid_forward(self, F, x, *args, **kwargs):
        # 1. Conv Block
        x = self.__conv(x)
        x = self.__bn(x)
        if self.__act is not None:
            x = self.__act(x)
        # 2. Down sample
        if self.__downsample is not None:
            x = self.__downsample(x)
        # Return
        return x
