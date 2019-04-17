
from config import Config
from dataLoader.dataset_flow import FlowDataset
from modellib.model import Model


video_dataset_name = 'hmdb'

# Configuration
cfg = Config(video_dataset=video_dataset_name, test_args=None)

# 1. Load dataset
video_dataset = FlowDataset(cfg, mode='train')

# 2. Create model
model = Model(cfg)

# 3. Train
model.train(video_dataset)
