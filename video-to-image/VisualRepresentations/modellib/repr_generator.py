
import os
import utils
import numpy as np
from model import Model
from mxnet import autograd, nd
from mxnet.gluon import nn, trainer
from basic_blocks import ConvBlock
from basic_utils.logger import get_logger


class AlexNet(Model):
    """
        AlexNet model from the `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    def __init__(self, cfg, **kwargs):
        super(AlexNet, self).__init__(cfg, prefix='repr_generator', **kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features')
            with self.features.name_scope():
                self.features.add(ConvBlock(64, kernel_size=11, strides=4, padding=2, activation='relu', downsample=True))
                self.features.add(ConvBlock(192, kernel_size=5, padding=2, activation='relu', downsample=True))
                self.features.add(ConvBlock(384, kernel_size=3, padding=1, activation='relu'))
                self.features.add(ConvBlock(256, kernel_size=3, padding=1, activation='relu'))
                self.features.add(ConvBlock(256, kernel_size=3, padding=1, activation='relu', downsample=True))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(self.cfg.NUM_CATEGORY['total'])

    def feedforward(self, x, mode):
        assert mode in ['train', 'generate']
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.feedforward(_x, mode) for _x in x]
            x = utils.io.gather_to_the_first_context(x)
            return x
        if mode == 'train':
            return self.output(self.features(x))
        else:
            return self.features(x)

    def train(self, image_train_dataset):
        # Get logger & params
        train_logger = get_logger(self.cfg.LOG_DIR, 'train_repr_generator')
        params_file, params_select = ['repr_generator'], ['repr_generator']
        # Init params
        self.load(params_file, params_select, train_logger, allow_init=True)
        # 1. Select params to train
        model_trainer = trainer.Trainer(self._collect_params(params_select), 'adam',
                                        {'wd': 5e-4, 'learning_rate': self.cfg.LR_INIT})
        # 2. Train each epoch
        for e in range(self.cfg.EPOCHS_TRAIN_GENERATOR):
            # Train each batch
            batch_index = 0
            while True:
                # 1. Load data
                (batch_images, batch_labels, _), finish = image_train_dataset.get_batch_data_cls(batch_index, self.cfg.BATCH_SIZE_TRAIN_GENERATOR)
                try: x_list, y = utils.io.split_and_load_gpu(self.cfg.CTX, [batch_images], batch_labels)
                except:
                    if finish: break
                    else: batch_index += 1
                    continue
                # 2. Record calculation
                with autograd.record():
                    pred_y = self.feedforward(x_list, mode='train')
                    loss_value = utils.loss.loss_softmax(pred_y, y)
                # 3. Backward & update
                loss_value.backward()
                nd.waitall()
                model_trainer.step(batch_size=self.cfg.BATCH_SIZE_TRAIN_GENERATOR)
                # Show info
                train_logger.info(self.get_loss_info(
                    'Train repr generator - ',
                    e, batch_index, image_train_dataset.num_data / self.cfg.BATCH_SIZE_TRAIN_GENERATOR,
                    loss_value, cls_pack=(pred_y, y)))
                # Move to next
                if finish: break
                else: batch_index += 1
            # Schedules
            self._step_update_learning_rate(e, model_trainer)
            self._step_save_params(e, params_file, params_select)
        # 3. Finish
        train_logger.info("Training accomplished. ")

    def generate(self, video_dataset):
        # Init params
        self.load(['repr_generator'], ['repr_generator'], None, allow_init=False)
        # Generate for each batch
        batch_index = 0
        while True:
            print("Generating label for video dataset - [%-5d/%-5d]..." % (batch_index, video_dataset.num_data / self.cfg.BATCH_SIZE_GENERATE))
            # 1. Load data
            (batch_images, _, batch_paths), finish = video_dataset.get_batch_data_cls(batch_index, self.cfg.BATCH_SIZE_GENERATE)
            x_list = utils.io.split_and_load_gpu(self.cfg.CTX, batch_images)
            # 2. Generate features
            with autograd.predict_mode():
                features = self.feedforward(x_list, mode='generate').asnumpy()
            # 3. Save
            for (cat_dir_name, cat_obj), feature in zip(batch_paths, features):
                cat_dir_path = os.path.join(self.cfg.VIDEO_REPR_DIR[video_dataset.dataset_info], cat_dir_name)
                if not os.path.exists(cat_dir_path): os.makedirs(cat_dir_path)
                np.save(os.path.join(cat_dir_path, os.path.splitext(cat_obj)[0] + '.npy'), feature[np.newaxis, :])
            # Move to next
            if finish: break
            else: batch_index += 1
        # Finish
        print("Generating accomplished. ")
