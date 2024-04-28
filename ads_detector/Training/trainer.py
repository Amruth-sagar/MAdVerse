import cv2
import os
import json
import math
import numpy as np
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
import wandb

setup_logger()

wandb.init(
    project="Third Run",
    config={
        "batch_size": 16,
        "learning_rate": 0.000125,
        "architecture": "Faster R-CNN",
        "dataset": "Ads-Dataset",
        "epochs": 50,
        "num_classes": 1,
        "backbone": "R50-FPN",
        "Region_of_Interests": "256",
        "Images_per_Batch": 8,
    }
)

register_coco_instances("ads_train_1", {}, "/scratch/train/instances_train.json", "/scratch/train/")
register_coco_instances("ads_val", {}, "/scratch/validation/instances_train.json", "/scratch/validation/")

cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file("./nn_config.yaml")
cfg.MODEL.WEIGHTS = "./model_final.pth"
cfg.DATASETS.TRAIN = ("ads_train_1",)
cfg.DATASETS.TEST = ("ads_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs("../model_weights_7/", exist_ok=True)
cfg.OUTPUT_DIR = '../model_weights_7/'

cfg.SOLVER.BASE_LR = 0.000125
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 8

model = build_model(cfg)

for name, parameters in model.named_parameters():
    if not name.startswith("roi_heads"):
        parameters.requires_grad_(False)
    if name.startswith("proposal_generator"):
        parameters.requires_grad_(True)

print("EPOCH 1")

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

output = trainer.storage.latest()
transformed_dict = {key: value[0] for key, value in output.items()}
wandb.log(dict(transformed_dict))

val = trainer.test(trainer.cfg, trainer.model, COCOEvaluator("ads_val", trainer.cfg, False, trainer.cfg.OUTPUT_DIR))
regular_dict = {key: value for key, value in val.items()}
wandb.log(dict(regular_dict))

epoch_num = 50
for i in range(0, epoch_num-1):
    print("EPOCH " + str(i+2))
    cfg.SOLVER.MAX_ITER = epoch*(i+2)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
    output = trainer.storage.latest()
    transformed_dict = {key: value[0] for key, value in output.items()}
    wandb.log(dict(transformed_dict))
    val = trainer.test(trainer.cfg, trainer.model, COCOEvaluator("ads_val", trainer.cfg, False, trainer.cfg.OUTPUT_DIR))
    regular_dict = {key: value for key, value in val.items()}
    wandb.log(dict(regular_dict))

wandb.finish()
