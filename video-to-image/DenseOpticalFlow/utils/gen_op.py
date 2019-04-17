
import numpy as np
from mxnet import nd
from basic_utils.common_layers import UpSample


def convert_to_image(y):
    """
    :param y: (n, 40, h, w)
    :return:
    """
    # 1. Calculate degrees.
    y = UpSample(y, 224, 224)
    degrees = np.arange(0, 2*np.pi, step=2*np.pi/40.0)
    degrees = np.sum(nd.softmax(y, axis=1).asnumpy() * degrees[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    # 2. Calculate x.
    x = (np.cos(degrees) + 1.0) / 2.0 * 255.0
    y = (np.sin(degrees) + 1.0) / 2.0 * 255.0
    z = np.zeros(shape=degrees.shape)
    img = np.asarray(np.concatenate([x[:, :, :, np.newaxis], y[:, :, :, np.newaxis], z[:, :, :, np.newaxis]], axis=3),
                     dtype='uint8')
    # Return
    return img

