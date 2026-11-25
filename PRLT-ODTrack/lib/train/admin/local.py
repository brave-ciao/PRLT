from pathlib import Path
import local_settings as local


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = str(Path(local.OUTPUT_DIR).expanduser().resolve())
        self.tensorboard_dir = str(Path(self.workspace_dir) / "tensorboard")
        self.pretrained_networks = str(local.PROJECT_ROOT / "pretrained_networks")
        self.lsotb_dir = str(local.LSOTB_TRAIN_DIR)
        self.lsotb_val_dir = str(local.LSOTB_VAL_DIR)
        for name in ["lasot", "got10k", "got10k_val", "trackingnet", "coco", "imagenet",
                     "lasot_lmdb", "got10k_lmdb", "trackingnet_lmdb", "coco_lmdb", "imagenet_lmdb",
                     "lvis", "sbd", "imagenetdet", "ecssd", "hkuis", "msra10k", "davis", "youtubevos"]:
            setattr(self, name + "_dir", "")
