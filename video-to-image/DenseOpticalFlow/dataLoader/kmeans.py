
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib


class KMeansCls(object):
    """
    KMeans classifier.
    """
    def __init__(self, cfg, video_dataset):
        self.cfg = cfg
        # 1. Create model
        self.__model = KMeans(n_clusters=40, n_jobs=4, max_iter=500)
        # 2. Load from existed file
        try:
            self.__load()
            return
        # 3. Train KMeans model.
        except:
            # (1) Collecting training samples.
            train_x = []
            for batch_index in range(self.cfg.BATCHES_USED_FOR_KMEANS_TRAINING):
                print("Collecting samples from %s for kmeans training [%-3d/%-3d]..." %
                      (video_dataset, batch_index, self.cfg.BATCHES_USED_FOR_KMEANS_TRAINING))
                _, batch_labels, finish = video_dataset.get_batch_data_cls(
                    batch_size=self.cfg.BATCH_SIZE, batch_index=batch_index, mode='train', train_usage='kmeans')
                train_x.append(batch_labels)
                if finish: break
            train_x = np.concatenate(train_x, axis=0)
            # (2) Train KMeans.
            print("Training KMeans... Number of samples: %-5d." % len(train_x))
            self.__model.fit(train_x)
            # Save
            self.__dump()

    def __load(self):
        self.__model = joblib.load(self.cfg.TRAIN_VIDEO_DATASET_KMEANS_PATH)

    def __dump(self):
        joblib.dump(self.__model, self.cfg.TRAIN_VIDEO_DATASET_KMEANS_PATH)

    def predict(self, x):
        return self.__model.predict(x)
