
import os
import pickle
import numpy as np
import graphviz as gv
from mxnet import nd
from mxnet.gluon.utils import split_and_load
from matplotlib import pyplot as plt

########################################################################################################################
# Load data & show info
########################################################################################################################


def load_pkl(file_path):
    f = open(file_path, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def save_pkl(file_path, obj):
    f = open(file_path, 'wb')
    pickle.dump(obj, f)
    f.close()


def shuffle_group(*args):
    """
    :param args:
    :return:
    """
    indices = np.arange(0, len(args[0]), step=1, dtype=np.int)
    np.random.shuffle(indices)
    # Shuffle
    r = [x[indices] for x in args]
    # Return
    return r


def split_and_load_gpu(ctx, *args):
    """
    :param ctx: List.
    :param args:
        for each 'a' in 'args':
            if isinstance(a, list): Then split 'a' to multi gpu.
            else: Load 'a' to the first gpu.
    :return:
    """
    assert isinstance(ctx, list)
    results = []
    for a in args:
        if isinstance(a, list):
            results.append(split_and_load(a[0], ctx_list=ctx, even_split=False))
        else:
            results.append(nd.array(a, ctx=ctx[0]))
    if len(results) == 1:
        results = results[0]
    return results


def gather_to_the_first_context(*args):
    result = [nd.concat(*[a.copyto(arg[0].context) for a in arg], dim=0) for arg in args]
    if len(result) == 1:
        result = result[0]
    return result


def get_logger_info_loss(loss_names, loss_value_list, mask_list=None):
    if mask_list is None: mask_list = [None for _ in loss_names]
    # 1. Init results
    info = ''
    # 2. Get results
    for name, loss_value, mask in zip(loss_names, loss_value_list, mask_list):
        if mask is None:
            metric = nd.mean(loss_value).asscalar()
            msg = '%s: %.5f. ' % (name, metric)
        else:
            denominator = nd.sum(mask).asscalar()
            if denominator > 0.00:
                metric = nd.sum(loss_value * mask).asscalar() / denominator
                msg = '%s: %.5f. ' % (name, metric)
            else:
                msg = '%s: None.    ' % name
        info += msg
    # Return
    return info


def get_cls_ap(pred, y, mode='argmax'):
    """
    :param pred: (batch, num_cat)
    :param y: (batch, num_cat)
    :param mode:
    :return:
    """
    assert mode in ['argmax', 'argmin']
    # Select function
    arg_func = nd.argmax if mode == 'argmax' else nd.argmin
    # Get accuracy.
    ap = nd.mean(arg_func(pred, axis=1) == arg_func(y, axis=1)).asscalar()
    return ap
