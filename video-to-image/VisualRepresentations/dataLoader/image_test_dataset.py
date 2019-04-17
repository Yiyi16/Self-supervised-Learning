
import os
import numpy as np
import mxnet as mx
from dataset import Dataset
from keras.utils import to_categorical


class ImageTestDataset(Dataset):
    """
    Image Dataset for testing.
    """
    def _generate_anns(self):
        # Determine image dataset name & mode
        dataset_name, dataset_mode = self.dataset_info
        assert (dataset_name in ['s40data', 'vocdata']) and (dataset_mode in ['train', 'val'])
        dataset_path = os.path.join(self.cfg.IMAGE_DATASET_DIR[dataset_name], dataset_mode)
        # Generate category name.
        category_names = []
        for cat_dir_name in os.listdir(dataset_path):
            if not os.path.isdir(os.path.join(dataset_path, cat_dir_name)): continue
            category_names.append(cat_dir_name)
        category_names.sort()
        # Generate annotations
        anns = []
        for cat_dir_name in os.listdir(dataset_path):
            cat_dir_path = os.path.join(dataset_path, cat_dir_name)
            if not os.path.isdir(cat_dir_path): continue
            for cat_obj in os.listdir(cat_dir_path):
                cat_obj_path = os.path.join(cat_dir_path, cat_obj)
                # Save
                anns.append([cat_dir_name, category_names.index(cat_dir_name), cat_obj_path])
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
