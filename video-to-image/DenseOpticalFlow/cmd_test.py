
from config import Config
from dataLoader.dataset_flow import FlowDataset
from modellib.model import Model


video_dataset_name = 'ucf101'
image_dataset = 's40data'
image_dataset_mode = 'train'


for video_dataset_name in ['ucf101', 'hmdb']:
    for image_dataset in ['s40data', 'vocdata']:
        for image_dataset_mode in ['train', 'val']:
            # Configuration
            cfg = Config(video_dataset=video_dataset_name, test_args=(image_dataset, image_dataset_mode))

            # 1. Load dataset
            img_dataset = FlowDataset(cfg, mode='test')

            # 2. Create model
            model = Model(cfg)

            # 3. Test
            model.test(img_dataset)
