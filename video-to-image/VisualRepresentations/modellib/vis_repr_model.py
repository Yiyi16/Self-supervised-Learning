
import os
import utils
import numpy as np
from mxnet import autograd, nd
from mxnet.gluon import nn, trainer
from model import Model
from modellib.basic_blocks import ConvBlock
from basic_utils.logger import get_logger


class VisReprModel(Model):
    """
    Network model.
    """
    def __init__(self, cfg, **kwargs):
        super(VisReprModel, self).__init__(cfg, prefix='vis_repr_model', **kwargs)
        # Configurations
        self.cfg = cfg
        # Network structure
        with self.name_scope():
            self._main = nn.HybridSequential(prefix='main')
            with self._main.name_scope():
                self._main.add(ConvBlock(channels=64, kernel_size=11, strides=4, padding=2, activation='relu', downsample=True))
                self._main.add(ConvBlock(channels=192, kernel_size=5, padding=2, activation='relu', downsample=True))
                self._main.add(ConvBlock(channels=384, kernel_size=3, padding=1, activation='relu'))
                self._main.add(ConvBlock(channels=256, kernel_size=3, padding=1, activation='relu'))
                self._main.add(ConvBlock(channels=256, kernel_size=3, padding=1, activation='relu', downsample=True))
                self._main.add(nn.Flatten())
                self._main.add(nn.Dense(4096, activation='relu'))
                self._main.add(nn.Dropout(0.5))
                self._main.add(nn.Dense(4096, activation='relu'))
                self._main.add(nn.Dropout(0.5))

########################################################################################################################
# Forward calculation
########################################################################################################################

    def feedforward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.feedforward(_x) for _x in x]
            x = utils.io.gather_to_the_first_context(x)
            return x
        return self._main(x)

########################################################################################################################
# Procedures
########################################################################################################################

    def train(self, video_dataset):
        # Get logger
        train_logger = get_logger(self.cfg.LOG_DIR, 'train_vis_repr_model_%s' % video_dataset.dataset_info)
        params_file, params_select = ['vis_repr_model_%s' % video_dataset.dataset_info], ['vis_repr_model']
        # Init params
        self.load(params_file, params_select, train_logger, allow_init=True)
        # 1. Select params to train
        model_trainer = trainer.Trainer(self._collect_params(params_select), 'adam',
                                        {'wd': 5e-4, 'learning_rate': self.cfg.LR_INIT})
        # 2. Train each epoch
        for e in range(self.cfg.EPOCHS_TRAIN_MAIN):
            # Train each batch
            batch_index = 0
            while True:
                # 1. Load data
                (batch_images, batch_labels, _), finish = video_dataset.get_batch_data_cls(batch_index, self.cfg.BATCH_SIZE_TRAIN_MAIN)
                x_list, y = utils.io.split_and_load_gpu(self.cfg.CTX, [batch_images], batch_labels)
                # 2. Record calculation
                with autograd.record():
                    pred_y = self.feedforward(x_list)
                    loss_value = utils.loss.loss_mse(pred_y, y)
                # 3. Backward & update
                loss_value.backward()
                nd.waitall()
                model_trainer.step(batch_size=self.cfg.BATCH_SIZE_TRAIN_MAIN)
                # Show info
                train_logger.info(self.get_loss_info(
                    'Train vis_repr_model - ', e, batch_index, video_dataset.num_data / self.cfg.BATCH_SIZE_TRAIN_MAIN,
                    loss_value))
                # Move to next
                if finish: break
                else: batch_index += 1
            # Schedules
            self._step_update_learning_rate(e, model_trainer)
            self._step_save_params(e, params_file, params_select)
        # 3. Finish
        train_logger.info("Training accomplished. ")

    def generate(self, image_dataset, video_dataset_name):
        # Init params
        self.load(['vis_repr_model_%s' % video_dataset_name], ['vis_repr_model'], None, allow_init=False)
        # Generate for each batch
        batch_index = 0
        while True:
            print("Generating label for image data - [%-5d/%-5d]..." % (batch_index, image_dataset.num_data / self.cfg.BATCH_SIZE_GENERATE))
            # 1. Load data
            (batch_images, _, batch_paths), finish = image_dataset.get_batch_data_cls(batch_index, self.cfg.BATCH_SIZE_GENERATE)
            x_list = utils.io.split_and_load_gpu(self.cfg.CTX, batch_images)
            # 2. Generate features
            with autograd.predict_mode():
                features = self.feedforward(x_list).asnumpy()
            # 3. Save
            for (cat_dir_name, cat_obj), feature in zip(batch_paths, features):
                cat_dir_path = os.path.join(
                    self.cfg.IMAGE_REPR_DIR, video_dataset_name,
                    image_dataset.dataset_info[0], image_dataset.dataset_info[1], cat_dir_name)
                if not os.path.exists(cat_dir_path): os.makedirs(cat_dir_path)
                np.save(os.path.join(cat_dir_path, os.path.splitext(cat_obj)[0] + '.npy'), feature)
            # Move to next
            if finish: break
            else: batch_index += 1
        # Finish
        print("Generating accomplished. ")

