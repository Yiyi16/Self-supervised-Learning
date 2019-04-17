
from mxnet.gluon import loss


loss_softmax = loss.SoftmaxCrossEntropyLoss(sparse_label=False)

loss_mse = loss.L2Loss()
