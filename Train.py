from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultTrainer
from torchvision.transforms import ToTensor
import Dataset


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("hw2_train",)
    cfg.OUTPUT_DIR = 'checkpoints'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00001
    trainer = DefaultTrainer(cfg)
    print('start')
    trainer.train()
    print('finish')
    