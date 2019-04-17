
import os
import utils
import numpy as np
import mxnet as mx
from imageio import imsave
from mxnet import nd, autograd
from mxnet.gluon import nn, trainer
from basic_blocks import ConvBlock
from basic_utils.logger import get_logger


class Model(nn.HybridBlock):
    """
    Attribute Network.
    """
    def __init__(self, cfg, **kwargs):
        super(Model, self).__init__(prefix='model', **kwargs)
        # Configuration
        self.cfg = cfg
        # Network structure
        with self.name_scope():
            self._conv1 = ConvBlock(channels=96, kernel_size=11, activation='relu', downsample=True)
            self._conv2 = ConvBlock(channels=256, kernel_size=5, activation='relu', downsample=True)
            self._conv3 = ConvBlock(channels=384, kernel_size=3, activation='relu')
            self._conv4 = ConvBlock(channels=384, kernel_size=3, activation='relu')
            self._conv5 = ConvBlock(channels=256, kernel_size=3, activation='relu', downsample=True)
            self._fc1 = nn.Dense(4096, activation='relu')
            self._fc2 = nn.Dense(4096, activation='relu')
            self._fc3 = nn.Dense(40)

########################################################################################################################
# Save & Load
########################################################################################################################

    def __collect_params(self, selects):
        """
        :param selects:
        :return:
        """
        params_selects = ''
        for s in selects: params_selects += (s + '+|')
        params_selects = params_selects[:-2]
        params = self.collect_params(params_selects)
        # Return
        return params

    def __load_params(self, selects, end, logger=None, allow_init=True):
        """
        :param selects:
        :param end:
        :param logger:
        :return:
        """
        assert len(selects) == len(end)
        for index, args in enumerate(zip(selects, end)):
            s, e = args
            # 1. Get stale params file info
            stale_params_file = self.cfg.params_path(s, e)
            # 2. Load from existed
            if os.path.exists(stale_params_file):
                self.collect_params(s + '+').load(stale_params_file, ctx=self.cfg.CTX)
            # 3. Init
            else:
                if e is not None: assert allow_init, 'Pre-trained params must be provided.'
                self.collect_params(s + '+').initialize(init=mx.initializer.Xavier(), ctx=self.cfg.CTX)
            # Echo
            if os.path.exists(stale_params_file):
                info = "Model %s params loaded from %s." % (s, stale_params_file)
            else:
                info = "Model %s params initialized from scratch." % s
            if logger is not None:
                logger.info(info)
            else:
                print(info)

    def __save_params(self, selects, end):
        """
        :param selects:
        :param end:
        :return:
        """
        for s, e in zip(selects, end):
            # 1. Remove stale params file
            params_file = self.cfg.params_path(s, end=e)
            if os.path.exists(params_file): os.remove(params_file)
            # 2. Collect params
            self.collect_params(s + '+').save(params_file)

    def save(self, params_selects, params_end):
        """
        :param params_selects:
        :param params_end:
        :return:
        """
        self.__save_params(params_selects, params_end if params_end is not None else [None for _ in params_selects])

    def load(self, params_selects, params_end=None, logger=None, allow_init=True):
        """
        :param params_selects:
        :param params_end:
        :param logger:
        :param allow_init:
        :return: Loaded from existed flag.
        """
        self.__load_params(params_selects, params_end if params_end is not None else [None for _ in params_selects],
                           logger, allow_init=allow_init)

########################################################################################################################
# Step Schedule
########################################################################################################################

    def __step_update_learning_rate(self, e, model_trainer):
        if (e % self.cfg.LR_STEP == 0) and (e > 0):
            model_trainer.set_learning_rate(model_trainer.learning_rate * self.cfg.LR_FACTOR)

    def __step_save_params(self, e, params_selects, params_end=None):
        if e % self.cfg.PARAMS_SAVE_STEP == 0:
            self.save(params_selects, params_end)

########################################################################################################################
# Forward calculation
########################################################################################################################

    def forward(self, x, *args):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.forward(_x) for _x in x]
            x = utils.io.gather_to_the_first_context(x)
            return x
        # (n, 3, h, w) -> (n, c, h', w')
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        # (n, c, h', w') -> (n*h'*w', c)
        n, c, h, w = x.shape
        x = nd.swapaxes(nd.reshape(nd.swapaxes(x, 0, 1), shape=(c, n*h*w)), 0, 1)
        # (n*h'*w', c) -> (n*h'*w', 40)
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        # (n*h'*w', 40) -> (n, 40, h', w')
        x = nd.swapaxes(nd.reshape(nd.swapaxes(x, 0, 1), shape=(40, n, h, w)), 0, 1)
        # Return
        return x

########################################################################################################################
# Train multi-gpu
########################################################################################################################

    def train(self, train_dataset):
        # Get logger
        train_logger = get_logger(self.cfg.LOG_DIR, 'train')
        # Init params
        self.load(['model'], logger=train_logger, allow_init=True)
        # 1. Select params to train
        params_select = ['model']
        model_trainer = trainer.Trainer(self.__collect_params(params_select), 'adam',
                                        {'wd': 5e-4, 'learning_rate': self.cfg.LR_INIT})
        # 2. Train each epoch
        for e in range(self.cfg.EPOCHS_TRAIN):
            # Train each batch
            batch_index = 0
            while True:
                # 1. Load data
                (batch_images, batch_labels), finish = train_dataset.get_batch_data_cls(
                    batch_size=self.cfg.BATCH_SIZE, batch_index=batch_index, mode='train', train_usage='flow')
                x_list, y = utils.io.split_and_load_gpu(self.cfg.CTX, [batch_images], batch_labels)
                # 2. Record calculation
                with autograd.record():
                    pred_y = self.forward(x_list)
                    loss_value = utils.loss.loss_softmax(pred_y, y)
                # 3. Backward & update
                loss_value.backward()
                nd.waitall()
                model_trainer.step(batch_size=self.cfg.BATCH_SIZE)
                # Show info
                train_logger.info(self.get_loss_info('Train - ', e, batch_index,
                                                     loss_value=loss_value, cls_pack=(pred_y, y)))
                # Move to next
                if finish: break
                else: batch_index += 1
            # Schedules
            self.__step_update_learning_rate(e, model_trainer)
            self.__step_save_params(e, params_select)
            # 3. Finish
        # 3. Finish
        train_logger.info("Training accomplished. ")

    def test(self, test_dataset):
        # Init params
        self.load(['model'], logger=None, allow_init=False)
        # Test
        # Train each batch
        batch_index = 0
        while True:
            print("Saving for batch [%-5d/%-5d]..." % (batch_index, test_dataset.num_data / self.cfg.BATCH_SIZE))
            # 1. Load data
            (batch_images, batch_labels, batch_paths), finish = test_dataset.get_batch_data_cls(
                batch_size=self.cfg.BATCH_SIZE, batch_index=batch_index, mode='test')
            x_list, y = utils.io.split_and_load_gpu(self.cfg.CTX, [batch_images], batch_labels)
            # 2. Record calculation
            with autograd.predict_mode():
                pred_y = self.forward(x_list)
                pred_y = utils.gen_op.convert_to_image(pred_y)
            # 3. Save to directory
            for py, (cat_dir_name, cur_obj) in zip(pred_y, batch_paths):
                # Make cat dir
                cat_dir = os.path.join(self.cfg.TEST_IMAGE_DATASET_FLOW_DIR, cat_dir_name)
                if not os.path.exists(cat_dir): os.makedirs(cat_dir)
                # Save
                imsave(os.path.join(cat_dir, cur_obj), py)
            # Move to next
            if finish: break
            else: batch_index += 1
        # Finish
        print("Testing accomplished. ")


########################################################################################################################
# Validation & Visualize
########################################################################################################################

    @staticmethod
    def get_loss_info(prefix, e, batch_index, loss_value, cls_pack):
        # 1. Prefix
        info = prefix
        if e is not None:
            if batch_index is not None:
                info += 'Epoch[%-3d], batch[%-3d]. ' % (e, batch_index)
            else:
                info += 'Epoch[%-3d]. ' % e
        else:
            if batch_index is not None: info += 'Batch[%-3d]. ' % batch_index
        # 2. Losses
        info += utils.io.get_logger_info_loss(['L_value'], [loss_value])
        # 3. Metric
        out, label = cls_pack
        n, c, h, w = out.shape
        out = nd.swapaxes(nd.reshape(nd.swapaxes(out, 0, 1), shape=(c, n*h*w)), 0, 1)
        label = nd.swapaxes(nd.reshape(nd.swapaxes(label, 0, 1), shape=(c, n*h*w)), 0, 1)
        info += 'Acc: %.5f. ' % utils.io.get_cls_ap(out, label)
        # Return
        return info
