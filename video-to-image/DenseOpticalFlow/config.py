
import os
import random
import mxnet as mx
import numpy as np
from mxnet import gpu


class Config(object):
    """
    Configurations
    """
    ####################################################################################################################
    # Datasets
    ####################################################################################################################
    # 1. Video datasets for training
    TRAIN_VIDEO_DATASET_DIR = None
    TRAIN_VIDEO_DATASET_ANN_PATH = None
    TRAIN_VIDEO_DATASET_KMEANS_PATH = None
    TRAIN_VIDEO_DATASET_FLOW_SAVE_DIR = None
    # 2. Image datasets for testing
    TEST_IMAGE_DATASET_DIR = '/home/panziqi/project/20190304_action_recognition/two-stream-pytorch/datasets'
    TEST_IMAGE_DATASET_FRAME_DIR = None
    TEST_IMAGE_DATASET_FLOW_DIR = None
    ####################################################################################################################
    # Setting
    ####################################################################################################################
    # 1. Paths
    ROOT_DIR = '/home/panziqi/project/20190304_action_recognition/DenseOpticalFlow'
    LOG_DIR = None
    PARAMS_DIR = None
    # 2. Data pre-process
    INPUT_SIZE = 200
    OUTPUT_SIZE = 18
    IMAGE_MEAN = mx.nd.array([0.485, 0.456, 0.406], dtype='float32')
    IMAGE_STD = mx.nd.array([0.229, 0.224, 0.225], dtype='float32')
    BATCHES_USED_FOR_KMEANS_TRAINING = 1
    # 3. Training
    # (1) Epochs
    EPOCHS_TRAIN = 5
    # (2) Batch size
    BATCH_SIZE = 1
    # (3) Learning rate
    LR_INIT = 0.0001
    LR_STEP = 30
    LR_FACTOR = 0.7
    # (4) Context
    CTX = [gpu(i) for i in [3]]
    # (5) Step
    PARAMS_SAVE_STEP = 1

    def __init__(self, video_dataset, test_args=None):
        assert video_dataset in ['ucf101', 'hmdb']
        # Paths
        self.LOG_DIR = os.path.join(self.ROOT_DIR, 'log', video_dataset)
        self.PARAMS_DIR = os.path.join(self.ROOT_DIR, 'params', video_dataset)
        self.TRAIN_VIDEO_DATASET_FLOW_SAVE_DIR = os.path.join(self.ROOT_DIR, 'datasets', video_dataset, 'saved_flow')
        self.TRAIN_VIDEO_DATASET_ANN_PATH = os.path.join(self.ROOT_DIR, 'datasets', video_dataset, '%s_annotations.json' % video_dataset)
        self.TRAIN_VIDEO_DATASET_KMEANS_PATH = os.path.join(self.PARAMS_DIR, '%s_kmeans.pkl' % video_dataset)
        if not os.path.exists(self.LOG_DIR): os.makedirs(self.LOG_DIR)
        if not os.path.exists(self.PARAMS_DIR): os.makedirs(self.PARAMS_DIR)
        if not os.path.exists(self.TRAIN_VIDEO_DATASET_FLOW_SAVE_DIR): os.makedirs(self.TRAIN_VIDEO_DATASET_FLOW_SAVE_DIR)
        # (1) For training.
        if test_args is None:
            self.TRAIN_VIDEO_DATASET_DIR = {
                'ucf101': '/home/zhangyy/UCF101/pyflow/flow_result',
                'hmdb': '/home/zhangyy/UCF101/pyflow/flow_result_HMDB'}[video_dataset]
        # (2) For testing.
        else:
            image_dataset, image_dataset_mode = test_args
            assert (image_dataset in ['s40data', 'vocdata']) and (image_dataset_mode in ['train', 'val'])
            self.TEST_IMAGE_DATASET_FRAME_DIR = os.path.join(self.TEST_IMAGE_DATASET_DIR, image_dataset, image_dataset_mode, 'frame')
            self.TEST_IMAGE_DATASET_FLOW_DIR = os.path.join(self.TEST_IMAGE_DATASET_DIR, image_dataset, image_dataset_mode, video_dataset, 'dense_optical_flow')
            if not os.path.exists(self.TEST_IMAGE_DATASET_FRAME_DIR): os.makedirs(self.TEST_IMAGE_DATASET_FRAME_DIR)
            if not os.path.exists(self.TEST_IMAGE_DATASET_FLOW_DIR): os.makedirs(self.TEST_IMAGE_DATASET_FLOW_DIR)

    def params_path(self, params_name, end=None):
        if end is not None: params_name += ('_' + end)
        params_file_path = os.path.join(self.PARAMS_DIR, '%s.params' % params_name)
        return params_file_path

    def proprec_image(self, image, mode):
        """
        :param image: (H, W, 3)
        :return: (1, 3, H, W)
        """
        flip_flag = None
        # Flip image
        if mode == 'train':
            if random.random() > 0.5:
                if random.random() > 0.5:
                    image = image[:, ::-1, :]
                    flip_flag = 1
                else:
                    image = image[::-1, :, :]
                    flip_flag = 0
        image = mx.image.imresize(image, w=self.INPUT_SIZE, h=self.INPUT_SIZE)
        image = image.astype('float32') / 255.0
        image = mx.image.color_normalize(image, mean=self.IMAGE_MEAN, std=self.IMAGE_STD)
        image = mx.nd.swapaxes(mx.nd.swapaxes(image, 1, 2), 0, 1)
        image = mx.nd.expand_dims(image, axis=0).asnumpy()
        # Return
        return image, flip_flag
