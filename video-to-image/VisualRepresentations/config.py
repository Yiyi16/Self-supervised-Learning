
import os
import mxnet as mx
from mxnet import gpu


class Config(object):
    """
    Configurations.
    """
    # 1. Directories.
    # (1) Dataset directories.
    IMAGE_DATASET_DIR = {
        's40data': '/home/panziqi/datasets/action_recognition/s40data',
        'vocdata': '/home/panziqi/datasets/action_recognition/vocdata'
    }
    VIDEO_DATASET_DIR = {
        'ucf101': '/home/panziqi/datasets/video_datasets/ucf101/small/frame',
        'hmdb': '/home/panziqi/datasets/video_datasets/hmdb/small/frame'
    }
    # (2) Project directories.
    ROOT_DIR = '/home/panziqi/project/20190304_action_recognition/VisualRepresentations/'
    LOG_DIR = os.path.join(ROOT_DIR, 'log')
    PARAMS_DIR = os.path.join(ROOT_DIR, 'params')
    VIDEO_DATASET_ANN_DIR = os.path.join(PARAMS_DIR, 'video_dataset_annotations')
    VIDEO_DATASET_ANN_PATH = {
        'ucf101': os.path.join(VIDEO_DATASET_ANN_DIR, 'ucf101.json'),
        'hmdb': os.path.join(VIDEO_DATASET_ANN_DIR, 'hmdb.json')
    }
    IMAGE_REPR_DIR = os.path.join(ROOT_DIR, 'visual_representations', 'image_datasets')
    VIDEO_REPR_DIR = {
        'ucf101': os.path.join(ROOT_DIR, 'visual_representations', 'video_datasets', 'ucf101'),
        'hmdb': os.path.join(ROOT_DIR, 'visual_representations', 'video_datasets', 'hmdb')
    }
    # 2. Data pre-process
    INPUT_SIZE = 224
    IMAGE_MEAN = mx.nd.array([0.485, 0.456, 0.406], dtype='float32')
    IMAGE_STD = mx.nd.array([0.229, 0.224, 0.225], dtype='float32')
    # 3. Settings.
    NUM_CATEGORY = {
        's40data': 40,
        'vocdata': 10,
        'total': 46
    }
    # (1) Epochs
    EPOCHS_TRAIN_GENERATOR = 50
    EPOCHS_TRAIN_MAIN = 5
    # (2) Batch size
    BATCH_SIZE_TRAIN_GENERATOR = 1024
    BATCH_SIZE_TRAIN_MAIN = 768
    BATCH_SIZE_GENERATE = 1
    # (3) Learning rate
    LR_INIT = 0.0001
    LR_STEP = 30
    LR_FACTOR = 0.7
    # (4) Context
    CTX = [gpu(i) for i in [2]]
    # (5) Step
    PARAMS_SAVE_STEP = 1

    def __init__(self):
        # Make directories.
        if not os.path.exists(self.LOG_DIR): os.makedirs(self.LOG_DIR)
        if not os.path.exists(self.PARAMS_DIR): os.makedirs(self.PARAMS_DIR)
        if not os.path.exists(self.VIDEO_DATASET_ANN_DIR): os.makedirs(self.VIDEO_DATASET_ANN_DIR)
        if not os.path.exists(self.IMAGE_REPR_DIR): os.makedirs(self.IMAGE_REPR_DIR)
        for _, repr_dir in self.VIDEO_REPR_DIR.items():
            if not os.path.exists(repr_dir): os.makedirs(repr_dir)

    def preproc_image(self, image):
        """
        :param image: (H, W, 3)
        :return: (1, 3, H, W)
        """
        image = mx.image.imresize(image, w=self.INPUT_SIZE, h=self.INPUT_SIZE)
        image = image.astype('float32') / 255.0
        image = mx.image.color_normalize(image, mean=self.IMAGE_MEAN, std=self.IMAGE_STD)
        image = mx.nd.swapaxes(mx.nd.swapaxes(image, 1, 2), 0, 1)
        image = mx.nd.expand_dims(image, axis=0).asnumpy()
        # Return
        return image

    def params_path(self, params_name):
        params_file_path = os.path.join(self.PARAMS_DIR, '%s.params' % params_name)
        return params_file_path


cfg = Config()
