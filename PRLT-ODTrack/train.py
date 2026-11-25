import argparse
import json
from pathlib import Path
import random

import local_settings as local


def configure_training(device):
    import cv2
    import numpy as np
    import torch

    torch.cuda.set_device(device)
    random.seed(local.SEED)
    np.random.seed(local.SEED)
    torch.manual_seed(local.SEED)
    torch.cuda.manual_seed(local.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(0)


def train(cfg, device, resume, pretrained, run_config):
    from torch.nn import BCEWithLogitsLoss
    from torch.nn.functional import l1_loss

    from lib.models.odtrack import build_odtrack
    from lib.train.admin.settings import Settings
    from lib.train.base_functions import get_optimizer_scheduler, update_settings
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss
    from prlt.actor import PRLTActor
    from prlt.data import build_dataloaders
    from prlt.io import strict_load_tracker, verify_full_finetuning
    from prlt.trainer import PRLTTrainer

    settings = Settings()
    settings.script_name = "odtrack"
    settings.config_name = local.CONFIG_NAME
    settings.project_path = "train/odtrack/" + local.CONFIG_NAME
    settings.save_dir = str(Path(local.OUTPUT_DIR).expanduser().resolve())
    settings.local_rank = -1
    settings.use_lmdb = False
    settings.use_wandb = False
    settings.skip_validation = not bool(local.LSOTB_VAL_DIR)
    settings.save_every_epoch = True
    settings.device = device
    settings.lsotb_protocol = run_config
    log_dir = Path(settings.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    settings.log_file = str(log_dir / f"odtrack-{local.CONFIG_NAME}.log")
    update_settings(settings, cfg)
    train_loader, val_loader = build_dataloaders(cfg, settings)

    model = build_odtrack(cfg)
    if not resume:
        strict_load_tracker(model, pretrained)
    model.to(device)
    objective = {"giou": giou_loss, "l1": l1_loss, "focal": FocalLoss(), "cls": BCEWithLogitsLoss()}
    weights = {"giou": cfg.TRAIN.GIOU_WEIGHT, "l1": cfg.TRAIN.L1_WEIGHT, "focal": 1.0, "cls": 1.0}
    actor = PRLTActor(model, objective, weights, settings, cfg)
    optimizer, scheduler = get_optimizer_scheduler(model, cfg, settings)
    verify_full_finetuning(model, optimizer)
    loaders = [train_loader] if val_loader is None else [train_loader, val_loader]
    trainer = PRLTTrainer(actor, loaders, optimizer, settings, scheduler, use_amp=cfg.TRAIN.AMP)
    trainer.train(cfg.TRAIN.EPOCH, load_latest=resume, fail_safe=False)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PRLT-ODTrack on LSOTB-TIR.")
    parser.add_argument("--data", type=Path, default=local.LSOTB_TRAIN_DIR)
    parser.add_argument("--pretrained", type=Path, default=local.PRETRAINED_TRACKER)
    parser.add_argument("--output", type=Path, default=local.OUTPUT_DIR)
    parser.add_argument("--gpu", type=int, default=local.GPU_ID)
    parser.add_argument("--seed", type=int, default=local.SEED)
    parser.add_argument("--resume", action="store_true", default=local.RESUME_TRAINING)
    args = parser.parse_args()

    import torch
    from prlt.io import checkpoint_directory, file_sha256, load_config, plain, reject_overlap
    from prlt.io import test_manifest_names, training_box_format, write_json
    from prlt.lsotb import load_split, read_rgb, split_fingerprint
    from prlt.options import protocol

    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("--gpu must select an available CUDA device")
    local.LSOTB_TRAIN_DIR = args.data.expanduser().resolve()
    local.OUTPUT_DIR = args.output.expanduser().resolve()
    local.SEED = args.seed
    cfg = load_config()
    directory = checkpoint_directory()
    existing = sorted(directory.glob("ODTrack_ep*.pth.tar"))
    if existing and not args.resume:
        raise FileExistsError(f"{directory} contains checkpoints. Use --resume or choose a new --output.")
    resume = bool(existing)
    pretrained = args.pretrained.expanduser().resolve()
    if not resume and not pretrained.is_file():
        raise FileNotFoundError(f"ODTrack initialization checkpoint not found: {pretrained}")

    device = torch.device(f"cuda:{args.gpu}")
    configure_training(device)
    records, excluded = load_split(local.LSOTB_TRAIN_DIR, training=True,
                                   box_format=training_box_format(), return_skipped=True)
    test_sequences = test_manifest_names()
    reject_overlap(records, test_sequences, "Training and LSOTB-TIR120")
    for record in records:
        read_rgb(record.frames[0])
        read_rgb(record.frames[-1])
    validation = []
    if local.LSOTB_VAL_DIR:
        validation = load_split(local.LSOTB_VAL_DIR, training=True,
                                box_format=training_box_format("val"), allow_mismatched_train=False)
        reject_overlap(validation, [r.name for r in records], "Training and validation")
        reject_overlap(validation, test_sequences, "Validation and LSOTB-TIR120")

    run_file = directory / "training_protocol.json"
    run_config = {
        "odtrack": plain(cfg),
        "prlt": protocol(),
        "seed": args.seed,
        "train_split": split_fingerprint(records),
        "train_annotation_format": training_box_format(),
        "excluded_train_sequences": excluded,
        "validation_split": split_fingerprint(validation) if validation else None,
        "validation_annotation_format": training_box_format("val") if validation else None,
        "initialization_sha256": file_sha256(pretrained) if pretrained.is_file() else None,
    }
    if resume:
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if run_config["initialization_sha256"] is None:
            run_config["initialization_sha256"] = previous["initialization_sha256"]
        if previous != run_config:
            raise ValueError("Resume configuration differs from the saved training protocol.")
    else:
        write_json(run_file, run_config)
    print(f"PRLT-ODTrack | seed {args.seed} | {len(records)} training sequences | checkpoints: {directory}", flush=True)
    train(cfg, device, resume, pretrained, run_config)


if __name__ == "__main__":
    main()
