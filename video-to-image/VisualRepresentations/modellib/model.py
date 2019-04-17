
import os
import utils
import mxnet as mx
from mxnet.gluon import nn


class Model(nn.HybridBlock):
    """
    Network model.
    """
    def __init__(self, cfg, **kwargs):
        super(Model, self).__init__(**kwargs)
        # Configurations
        self.cfg = cfg

########################################################################################################################
# Save & Load
########################################################################################################################

    def _collect_params(self, selects):
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

    def load(self, file_names, selects, logger=None, allow_init=True):
        """
        :param selects:
        :param end:
        :param logger:
        :return:
        """
        assert len(selects) == len(file_names)
        for index, (file_name, select) in enumerate(zip(file_names, selects)):
            # 1. Get stale params file info
            stale_params_file = self.cfg.params_path(file_name)
            # 2. Load from existed
            if os.path.exists(stale_params_file):
                self.collect_params(select + '+').load(stale_params_file, ctx=self.cfg.CTX)
            # 3. Init
            else:
                assert allow_init, 'Pre-trained params must be provided.'
                self.collect_params(select + '+').initialize(init=mx.initializer.Xavier(), ctx=self.cfg.CTX)
            # Echo
            if os.path.exists(stale_params_file):
                info = "Model %s params loaded from %s." % (select, stale_params_file)
            else:
                info = "Model %s params initialized from scratch." % select
            if logger is not None:
                logger.info(info)
            else:
                print(info)

    def save(self, file_names, selects):
        """
        :param selects:
        :param end:
        :return:
        """
        for file_name, select in zip(file_names, selects):
            # 1. Remove stale params file
            params_file = self.cfg.params_path(file_name)
            if os.path.exists(params_file): os.remove(params_file)
            # 2. Collect params
            self.collect_params(select + '+').save(params_file)

########################################################################################################################
# Step Schedule
########################################################################################################################

    def _step_update_learning_rate(self, e, model_trainer):
        if (e % self.cfg.LR_STEP == 0) and (e > 0):
            model_trainer.set_learning_rate(model_trainer.learning_rate * self.cfg.LR_FACTOR)

    def _step_save_params(self, e, file_names, params_selects):
        if e % self.cfg.PARAMS_SAVE_STEP == 0:
            self.save(file_names, params_selects)

########################################################################################################################
# Validation & Visualize
########################################################################################################################

    @staticmethod
    def get_loss_info(prefix, e, batch_index, batch_count, loss_value=None, cls_pack=None):
        # 1. Prefix
        info = prefix
        if e is not None:
            if batch_index is not None:
                info += 'Epoch[%-3d], batch[%-3d/%-3d]. ' % (e, batch_index, batch_count)
            else:
                info += 'Epoch[%-3d]. ' % e
        else:
            if batch_index is not None: info += 'Batch[%-3d/%-3d]. ' % (batch_index, batch_count)
        # 2. Losses
        if loss_value is not None:
            info += utils.io.get_logger_info_loss(['L_value'], [loss_value])
        # 3. Metric
        if cls_pack is not None:
            out, label = cls_pack
            info += 'Acc: %.5f. ' % utils.io.get_cls_ap(out, label)
        # Return
        return info
