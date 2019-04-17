import os
import json
import random
import numpy as np
import mxnet as mx
from dataset import Dataset


class VideoDataset(Dataset):
    """
    Dataset for video.
    """
    def _generate_anns(self):
        # Get paths
        video_dataset_ann_path = self.cfg.VIDEO_DATASET_ANN_PATH[self.dataset_info]
        video_dataset_dir_path = self.cfg.VIDEO_DATASET_DIR[self.dataset_info]
        # Try to load from existed file.
        if os.path.exists(video_dataset_ann_path):
            f = open(video_dataset_ann_path, 'r')
            anns = json.load(f)
            f.close()
            return anns
        # Generate annotations
        anns = []
        for index, cat_dir_name in enumerate(os.listdir(video_dataset_dir_path)):
            print("Generating annotations for [%-5d/%-5d] - [%s]..." % (index, len(os.listdir(video_dataset_dir_path)), cat_dir_name))
            cat_dir_path = os.path.join(video_dataset_dir_path, cat_dir_name)
            if not os.path.isdir(cat_dir_path): continue
            for cat_obj in os.listdir(cat_dir_path):
                # Save
                anns.append([cat_dir_name, 0, os.path.join(cat_dir_path, cat_obj)])
        # Save annotations
        f = open(video_dataset_ann_path, 'w')
        json.dump(anns, f)
        f.close()
        # Return
        return anns

    def _get_batch_data_cls_from_indices(self, indices, **kwargs):
        # Init results
        batch_images, batch_labels, batch_paths = [], [], []
        # Load batch data
        for index in indices:
            # Split path
            cat_dir_name, cat_obj = str(self._anns[index][2]).split('/')[-2:]
            # 1. Load image
            image = self.cfg.preproc_image(mx.image.imread(self._anns[index][2]))
            batch_images.append(image)
            # 2. Load label
            try:
                # Random choice
                cat_dir_path = os.path.join(self.cfg.VIDEO_DATASET_DIR[self.dataset_info], cat_dir_name)
                cat_obj_name = os.path.splitext(os.listdir(cat_dir_path)[random.randint(0, len(os.listdir(cat_dir_path)) - 1)])[0]
                # Load & save
                label = np.load(os.path.join(self.cfg.VIDEO_REPR_DIR[self.dataset_info], cat_dir_name, cat_obj_name + '.npy'))
                batch_labels.append(label)
            except:
                pass
            # 3. Load paths.
            batch_paths.append([cat_dir_name, cat_obj])
        # Concat
        batch_images = np.concatenate(batch_images, axis=0)
        if batch_labels: batch_labels = np.concatenate(batch_labels, axis=0)
        assert len(batch_images) == len(batch_labels)
        # Return
        return batch_images, batch_labels, batch_paths
