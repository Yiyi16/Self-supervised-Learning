
import os
import json
import kmeans
import dataset
import numpy as np
import mxnet as mx
from skimage.transform import resize
from keras.utils import to_categorical


class FlowDataset(dataset.Dataset):
    """
    Optical Flow dataset.
    """
    def __init__(self, cfg, mode):
        super(FlowDataset, self).__init__(cfg, mode)
        # Create KMeans classifier for training
        if mode == 'train':
            self._kmeans_cls = kmeans.KMeansCls(cfg, self)

    def _generate_anns(self, mode):
        # Generate annotations
        anns = []
        # 1. For mode == 'train', generate annotations for video dataset. (cat_name, cat_index, file_path(without ext))
        if mode == 'train':
            # Try to load from existed file.
            if os.path.exists(self.cfg.TRAIN_VIDEO_DATASET_ANN_PATH):
                f = open(self.cfg.TRAIN_VIDEO_DATASET_ANN_PATH, 'r')
                anns = json.load(f)
                f.close()
                return anns
            # Generate from file
            # (1) Container for check duplicate
            chk_cat_container = []
            # (2) Load each directory
            for cat_index, cat_dir_name in enumerate(os.listdir(self.cfg.TRAIN_VIDEO_DATASET_DIR)):
                # Get cat name
                cat_name = str.split(cat_dir_name, '_')[1]
                if cat_name not in chk_cat_container: chk_cat_container.append(cat_name)
                # Get cat_dir_path
                cat_dir_path = os.path.join(self.cfg.TRAIN_VIDEO_DATASET_DIR, cat_dir_name)
                # Show progress
                if cat_index % 5 == 0:
                    print("Generating annotations for category[%-3d / %-3d][%-3d - %s]..." %
                          (cat_index, len(os.listdir(self.cfg.TRAIN_VIDEO_DATASET_DIR)),
                           chk_cat_container.index(cat_name), cat_dir_name))
                # (1) Container for check duplicate under cat directory
                chk_file_container = []
                # (2) Load each file
                for cat_obj in os.listdir(cat_dir_path):
                    cat_obj_name, cat_obj_ext = os.path.splitext(cat_obj)
                    # Check
                    if cat_obj_name in chk_file_container: continue
                    if not os.path.exists(os.path.join(cat_dir_path, '%s.jpg' % cat_obj_name)): continue
                    try: np.load(os.path.join(cat_dir_path, '%s.npy' % cat_obj_name))
                    except: continue
                    # Save
                    anns.append([cat_name, chk_cat_container.index(cat_name), os.path.join(cat_dir_path, cat_obj_name)])
                    chk_file_container.append(cat_obj_name)
            # Save to file
            f = open(self.cfg.TRAIN_VIDEO_DATASET_ANN_PATH, 'w')
            json.dump(anns, f)
            f.close()
        # 2. For mode == 'test', generate annotations for image dataset. (cat_name, cat_index, file_path)
        else:
            for cat_index, cat_dir_name in enumerate(os.listdir(self.cfg.TEST_IMAGE_DATASET_FRAME_DIR)):
                print("Generating annotations for category [%-3d - '%s']..." % (cat_index, cat_dir_name))
                cat_dir_path = os.path.join(self.cfg.TEST_IMAGE_DATASET_FRAME_DIR, cat_dir_name)
                for cat_obj in os.listdir(cat_dir_path):
                    anns.append([cat_dir_name, cat_index, os.path.join(cat_dir_path, cat_obj)])
        # Return
        return anns

    def _load_data(self, file_path, mode):
        image = mx.image.imread(file_path)
        image, flip_flag = self.cfg.proprec_image(image, mode)
        return image, flip_flag

    def _get_batch_data_cls_from_indices(self, indices, **kwargs):
        assert ('mode' in kwargs.keys()) and (kwargs['mode'] in ['train', 'test'])
        # Init results
        batch_images = []
        batch_labels = []
        batch_paths = []
        # 1. For test
        if kwargs['mode'] == 'test':
            # 1. Process data
            for index in indices:
                # 1. Get image. (1, 3, H, W)
                image, _ = self._load_data(self._anns[index][2], kwargs['mode'])
                # 2. Get label. Scalar.
                label = self._anns[index][1]
                # Save
                batch_images.append(image)
                batch_labels.append(label)
                batch_paths.append(str(self._anns[index][2]).split('/')[-2:])
            # 2. Convert to numpy & category
            batch_images = np.concatenate(batch_images, axis=0)
            batch_labels = np.array(batch_labels, dtype=np.int32)
            batch_labels = to_categorical(batch_labels, num_classes=self.num_category)
            # Return
            return batch_images, batch_labels, batch_paths
        # 2. For train
        else:
            assert ('train_usage' in kwargs.keys()) and (kwargs['train_usage'] in ['kmeans', 'flow'])
            # 1. For training kmeans
            if kwargs['train_usage'] == 'kmeans':
                for index in indices:
                    # 1. Get flow. (H, W, 2)
                    flow = np.load(self._anns[index][2] + '.npy')
                    # 2. Reshape to (H*W, 2)
                    flow = np.reshape(flow, newshape=(-1, 2))
                    # Save to labels.
                    batch_labels.append(flow)
                batch_labels = np.concatenate(batch_labels, axis=0)
            # 2. For training optical flow
            else:
                for index in indices:
                    # 1. Get image. (1, 3, H, W)
                    image, flip_flag = self._load_data(self._anns[index][2] + '.jpg', kwargs['mode'])
                    # 2. Get flow.
                    cat_dir_name, file_name = str(self._anns[index][2]).split('/')[-2:]
                    cat_dir_path = os.path.join(self.cfg.TRAIN_VIDEO_DATASET_FLOW_SAVE_DIR, cat_dir_name)
                    file_path = os.path.join(cat_dir_path, file_name + '.npy')
                    # Try to load from file.
                    try:
                        flow = np.load(file_path)
                    except:
                        # (1) Get original flow. (2, H, W)
                        flow = np.load(self._anns[index][2] + '.npy')
                        flow = np.swapaxes(np.swapaxes(flow, 1, 2), 0, 1)
                        # (2) Resize flow. (2, H, W)
                        flow_x = resize(flow[0, :, :], output_shape=(self.cfg.OUTPUT_SIZE, self.cfg.OUTPUT_SIZE))
                        flow_y = resize(flow[1, :, :], output_shape=(self.cfg.OUTPUT_SIZE, self.cfg.OUTPUT_SIZE))
                        flow = np.concatenate([flow_x[np.newaxis, :, :], flow_y[np.newaxis, :, :]], axis=0)
                        # (3) Flip. (2, H, W)
                        if flip_flag == 0:
                            flow = flow[:, ::-1, :]
                        elif flip_flag == 1:
                            flow = flow[:, :, ::-1]
                        # (4) Use KMeans. (40, H, W) ->
                        _, height, width = flow.shape
                        flow = np.reshape(flow, newshape=(2, height * width))
                        flow = self._kmeans_cls.predict(np.swapaxes(flow, 0, 1))
                        flow = to_categorical(flow, num_classes=40)
                        flow = np.reshape(np.swapaxes(flow, 0, 1), newshape=(1, 40, height, width))
                        # Save to file
                        if not os.path.exists(cat_dir_path): os.makedirs(cat_dir_path)
                        np.save(file_path, flow)
                    # Save
                    batch_images.append(image)
                    batch_labels.append(flow)
                batch_images = np.concatenate(batch_images, axis=0)
                batch_labels = np.concatenate(batch_labels, axis=0)
            # Return
            return batch_images, batch_labels
