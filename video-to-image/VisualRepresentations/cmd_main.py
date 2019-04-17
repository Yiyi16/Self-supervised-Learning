
import sys
from config import cfg
from dataLoader.image_train_dataset import ImageTrainDataset
from dataLoader.image_test_dataset import ImageTestDataset
from dataLoader.video_dataset import VideoDataset
from modellib.repr_generator import AlexNet
from modellib.vis_repr_model import VisReprModel


stage = int(sys.argv[1])

# 1. Train repr_generator.
if stage == 1:
    image_train_dataset = ImageTrainDataset(cfg)
    repr_generator = AlexNet(cfg)
    repr_generator.train(image_train_dataset)
# 2. Generate vis repr for video dataset
elif stage == 2:
    for video_dataset_name in ['ucf101', 'hmdb']:
        video_dataset = VideoDataset(cfg, dataset_info=video_dataset_name)
        repr_generator = AlexNet(cfg)
        repr_generator.generate(video_dataset)
# 3. Train vis repr model
elif stage == 3:
    for video_dataset_name in ['ucf101', 'hmdb']:
        video_dataset = VideoDataset(cfg, dataset_info=video_dataset_name)
        vis_repr_model = VisReprModel(cfg)
        vis_repr_model.train(video_dataset)
# 4. Generate vis repr for images
else:
    for video_dataset_name in ['ucf101', 'hmdb']:
        for image_dataset_name in ['s40data', 'vocdata']:
            for image_dataset_mode in ['train', 'val']:
                image_test_dataset = ImageTestDataset(cfg, dataset_info=(image_dataset_name, image_dataset_mode))
                vis_repr_model = VisReprModel(cfg)
                vis_repr_model.generate(image_test_dataset, video_dataset_name)
