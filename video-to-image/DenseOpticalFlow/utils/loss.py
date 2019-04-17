
from mxnet.gluon import loss


loss_softmax = loss.SoftmaxCrossEntropyLoss(sparse_label=False, axis=1)
