
import numpy as np
from keras.utils import to_categorical


class Dataset(object):
    """
    Abstract dataset.
    """
    def __init__(self, cfg, mode):
        # Configurations
        self.cfg = cfg
        # 1. Annotations. List of (cat_marker, cat_index, file_path)
        if isinstance(mode, str):
            assert mode in ['train', 'test']
            self._anns = self._generate_anns(mode)
        else:
            self._anns = mode
        # 2. Indices
        # (1) For common classification
        self._common_indices = {}
        # (2) For few-shot classification
        self._fewshot_indices = None

    def _generate_anns(self, mode):
        raise NotImplementedError

    def _load_data(self, file_path, mode):
        raise NotImplementedError

    @staticmethod
    def __calc_batch_indices(batch_index, num_data, batch_size):
        """
        :param batch_index:
        :param num_data
        :param batch_size
        :return:
        """
        # (1) Calculate original indices for current batch
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size
        flag_batch_end = False
        # (2) If overflow, adjust indices and set self._batch_index to -1 so next batch will terminate the iteration.
        if end_index >= num_data:
            end_index = num_data
            flag_batch_end = True
        # Return
        return start_index, end_index, flag_batch_end

########################################################################################################################
# Properties
########################################################################################################################

    @property
    def num_category(self):
        return max([i[1] for i in self._anns]) + 1

    @property
    def num_data(self):
        return len(self._anns)

    @property
    def category_names(self):
        """
        Category names.
        :return: List of category name.
        """
        names = [None for _ in range(self.num_category)]
        for key, key_index, _ in self._anns:
            names[key_index] = key
        # Return
        return names

    @property
    def category_indices(self):
        label_indicators = np.array([i[1] for i in self._anns], dtype=np.int32)
        basic_indices = [np.argwhere(label_indicators == i)[:, 0] for i in range(self.num_category)]
        return basic_indices

########################################################################################################################
# Get batch data for common classification
########################################################################################################################

    def __prepare_cls_indices(self, selects=None):
        # 1. Get basic indices. List of (num_cat, ), each is (num_cat_examples, )
        basic_indices = self.category_indices
        # 2. SELECTIONS
        if selects is not None:
            basic_indices = [basic_indices[s] for s in selects]
        # 3. Concat
        basic_indices = np.concatenate(basic_indices, axis=0)
        np.random.shuffle(basic_indices)
        # Return
        return basic_indices

    def _get_batch_data_cls_from_indices(self, indices, **kwargs):
        raise NotImplementedError

    def get_batch_data_cls(self, batch_size=None, batch_index=None, selects=None, **kwargs):
        """
        :param batch_size:
        :param batch_index:
        :param selects:
        :param preproc:
        :return:
        """
        # Calculate data indices
        if str(selects) not in self._common_indices.keys():
            self._common_indices.update({str(selects): self.__prepare_cls_indices(selects)})
        indices = self._common_indices[str(selects)]
        # 1. Calculate current batch's start & end indices
        if batch_size is None and batch_index is None: batch_size, batch_index = len(indices), 0
        assert len(indices) % batch_size == 0, 'Batch size is inappropriate, it should divide %d.' % len(indices)
        start_index, end_index, finish = self.__calc_batch_indices(batch_index, len(indices), batch_size)
        # 2. Get batch data
        batch_data = self._get_batch_data_cls_from_indices(indices[start_index:end_index], **kwargs)
        # Return
        return batch_data, finish
