import os

from lib.utils.box_ops import giou_loss, iouhead_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss


from lib.train.trainers import LTRTrainer

from torch.nn.parallel import DistributedDataParallel as DDP

from .base_functions import *

from lib.models.odtrack import build_odtrack

from lib.train.actors import ODTrackActor

import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'


    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')


    update_settings(settings, cfg)

    if settings.config_name.startswith("lsotb_") and not (
            getattr(settings, "init_tracker_checkpoint", None) or getattr(settings, "resume_training", False)):
        raise RuntimeError("LSOTB full-network fine-tuning requires explicit full-checkpoint initialization or "
                           "a validated resume. Use the project-root run_01_train.py entrypoint.")


    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))


    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir


    if settings.script_name == "odtrack":
        net = build_odtrack(cfg)
    else:
        raise ValueError("illegal script name")

    if getattr(settings, "init_tracker_checkpoint", None):
        from prlt.io import strict_load_tracker, write_json
        report = strict_load_tracker(net, settings.init_tracker_checkpoint)
        if settings.local_rank in [-1, 0]:
            write_json(os.path.join(settings.save_dir, "initialization_report.json"), report)


    net.cuda()
    if settings.local_rank != -1:

        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")


    if settings.script_name == "odtrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.0, 'cls': 1.0}
        actor = ODTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")


    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg, settings)
    if settings.config_name.startswith("lsotb_"):
        from prlt.io import verify_full_finetuning, write_json
        finetuning_report = verify_full_finetuning(net, optimizer)
        if settings.local_rank in [-1, 0]:
            write_json(os.path.join(settings.save_dir, "finetuning_report.json"), finetuning_report)
            print("Full ODTrack fine-tuning: backbone and head unfrozen; all parameters in optimizer.")
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train] + ([loader_val] if loader_val is not None else []), optimizer, settings, lr_scheduler, use_amp=use_amp)


    trainer.train(getattr(settings, "stop_after_epoch", cfg.TRAIN.EPOCH), load_latest=getattr(settings, "resume_training", True),
                  fail_safe=not settings.config_name.startswith("lsotb_"))
