
import os
import numpy as np
import mxnet as mx
from dataset import Dataset
from keras.utils import to_categorical


class ImageTrainDataset(Dataset):
    """
    Image Dataset for training.
    """
    def _generate_anns(self):
        # Get paths
        s40_dataset_train_path = os.path.join(self.cfg.IMAGE_DATASET_DIR['s40data'], 'train')
        s40_dataset_val_path = os.path.join(self.cfg.IMAGE_DATASET_DIR['s40data'], 'val')
        voc_dataset_train_path = os.path.join(self.cfg.IMAGE_DATASET_DIR['vocdata'], 'train')
        voc_dataset_val_path = os.path.join(self.cfg.IMAGE_DATASET_DIR['vocdata'], 'val')
        # Generate annotations
        annotations = {}
        for dataset_path in [s40_dataset_train_path, s40_dataset_val_path, voc_dataset_train_path, voc_dataset_val_path]:
            for cat_dir_name in os.listdir(dataset_path):
                # Save category
                if cat_dir_name not in annotations.keys(): annotations.update({cat_dir_name: []})
                # Collect category samples.
                cat_dir_path = os.path.join(dataset_path, cat_dir_name)
                if not os.path.isdir(cat_dir_path): continue
                for cat_obj in os.listdir(cat_dir_path):
                    annotations[cat_dir_name].append(os.path.join(cat_dir_path, cat_obj))
        # To lists.
        # (1) Sort category names.
        category_names = list(annotations.keys())
        category_names.sort()
        # (2) Generate
        anns = []
        for cat_name in annotations.keys():
            for cat_obj_path in annotations[cat_name]:
                anns.append([cat_name, category_names.index(cat_name), cat_obj_path])
        # Return
        return anns

    def _get_batch_data_cls_from_indices(self, indices, **kwargs):
        # Init results
        batch_images, batch_labels, batch_paths = [], [], []
        # Load batch data
        for index in indices:
            # 1. Load image
            image = self.cfg.preproc_image(mx.image.imread(self._anns[index][2]))
            batch_images.append(image)
            # 2. Load label
            batch_labels.append(self._anns[index][1])
            # 3. Load paths
            batch_paths.append(str(self._anns[index][2]).split('/')[-2:])
        # Concat
        batch_images = np.concatenate(batch_images, axis=0)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_labels = to_categorical(batch_labels, num_classes=self.num_category)
        # Return
        return batch_images, batch_labels, batch_paths
