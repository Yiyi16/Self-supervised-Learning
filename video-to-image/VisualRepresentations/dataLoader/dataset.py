
import numpy as np


class Dataset(object):
    """
    Abstract dataset.
    """
    def __init__(self, cfg, dataset_info=None):
        # Configurations
        self.cfg = cfg
        self._dataset_info = dataset_info
        # 1. Annotations. List of (cat_name, cat_index, file_path)
        self._anns = self._generate_anns()
        # 2. Indices
        self._indices = None

    def _generate_anns(self):
        raise NotImplementedError

    def __calc_batch_indices(self, batch_index, batch_size):
        # 1. Calculate original indices for current batch data
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size
        # 2. For case overflow
        if end_index >= self.num_data:
            end_index = self.num_data
            flag_batch_end = True
        else:
            flag_batch_end = False
        # Return
        return start_index, end_index, flag_batch_end

########################################################################################################################
# Properties
########################################################################################################################

    @property
    def dataset_info(self):
        return self._dataset_info

    @property
    def num_category(self):
        return max(set(i[1] for i in self._anns)) + 1

    @property
    def num_data(self):
        return len(self._anns)

    @property
    def category_names(self):
        names = [None for _ in range(self.num_category)]
        for key, key_index, _ in self._anns:
            names[key_index] = key
        return names

    @property
    def category_indices(self):
        label_indicators = np.array([i[1] for i in self._anns], dtype=np.int32)
        category_indices = [np.argwhere(label_indicators == i)[:, 0] for i in range(self.num_category)]
        return category_indices

########################################################################################################################
# Get batch data for common classification
########################################################################################################################

    def __prepare_cls_indices(self):
        category_indices = np.concatenate(self.category_indices, axis=0)
        np.random.shuffle(category_indices)
        return category_indices

    def _get_batch_data_cls_from_indices(self, indices, **kwargs):
        raise NotImplementedError

    def get_batch_data_cls(self, batch_index, batch_size, **kwargs):
        # Prepare indices.
        if self._indices is None: self._indices = self.__prepare_cls_indices()
        # 1. Calculate current batch' start & end indices
        start_index, end_index, finish = self.__calc_batch_indices(batch_index, batch_size)
        # 2. Get batch data
        batch_data = self._get_batch_data_cls_from_indices(self._indices[start_index:end_index], **kwargs)
        # Return
        return batch_data, finish
