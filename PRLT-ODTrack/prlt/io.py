from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

import local_settings as local


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def training_box_format(split="train"):


    default = getattr(local, "LSOTB_TRAIN_BOX_FORMAT", "xyxy")
    if split == "train":
        value = default
    elif split == "val":
        value = getattr(local, "LSOTB_VAL_BOX_FORMAT", default)
    else:
        raise ValueError("This option applies only to train/val, not test predictions")
    value = str(value).strip().lower()
    if value not in ("xyxy", "xywh"):
        raise ValueError("LSOTB annotation format must be 'xyxy' or 'xywh', got {!r}".format(value))
    return value


def load_config():
    from lib.config.odtrack.config import cfg as defaults, update_config_from_file
    cfg = deepcopy(defaults)
    path = local.PROJECT_ROOT / "experiments/odtrack" / (local.CONFIG_NAME + ".yaml")
    update_config_from_file(str(path), base_cfg=cfg)
    if cfg.MODEL.BACKBONE.TYPE != "vit_base_patch16_224" or cfg.MODEL.BACKBONE.CE_LOC or cfg.MODEL.BACKBONE.CE_KEEP_RATIO:
        raise ValueError("This experiment requires the official non-CE ViT-B/16 branch")
    if not cfg.MODEL.BACKBONE.ADD_CLS_TOKEN:
        raise ValueError("ODTrack temporal token must be enabled")
    if (cfg.DATA.SEARCH.SIZE, cfg.DATA.TEMPLATE.SIZE, cfg.TEST.SEARCH_SIZE, cfg.TEST.TEMPLATE_SIZE) != (384, 192, 384, 192):
        raise ValueError("This experiment requires search=384, template=192 in both train and test")
    if (cfg.DATA.TEMPLATE.NUMBER, cfg.DATA.SEARCH.NUMBER) != (3, 2):
        raise ValueError("Official training uses three templates and two searches")
    return cfg


def verify_full_finetuning(model, optimizer=None):

    core = getattr(model, "module", model)
    parameters = dict(core.named_parameters())
    frozen = [name for name, parameter in parameters.items() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError("Full ODTrack fine-tuning requires unfrozen parameters: " + repr(frozen))
    if not any(name.startswith("backbone.") for name in parameters) or not any(
            name.startswith("box_head.") for name in parameters):
        raise RuntimeError("Expected the original ODTrack backbone and box head")
    report = {
        "training_mode": "full_finetuning",
        "frozen_parameter_names": frozen,
        "parameter_tensors": len(parameters),
        "total_parameters": sum(p.numel() for p in parameters.values()),
        "backbone_parameters": sum(p.numel() for n, p in parameters.items() if n.startswith("backbone.")),
        "head_parameters": sum(p.numel() for n, p in parameters.items() if n.startswith("box_head.")),
    }
    if optimizer is not None:
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        expected_ids = {id(p) for p in parameters.values()}
        actual_ids = {id(p) for p in optimized}
        missing = [name for name, p in parameters.items() if id(p) not in actual_ids]
        if missing or actual_ids != expected_ids or len(optimized) != len(actual_ids):
            raise RuntimeError("Optimizer must contain every original parameter exactly once; missing=" + repr(missing))
        report["optimizer_covers_all_parameters"] = True
        report["optimizer_groups"] = [{"lr": float(group["lr"]),
                                       "parameter_tensors": len(group["params"]),
                                       "parameters": sum(p.numel() for p in group["params"])}
                                      for group in optimizer.param_groups]
    return report


def trusted_torch_load(path):

    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def strict_load_tracker(model, checkpoint_path):
    import torch
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Missing full ODTrack checkpoint: {}".format(checkpoint_path))
    checkpoint = trusted_torch_load(checkpoint_path)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state = checkpoint["net"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError("Expected full ODTrack weights ('net'), not MAE backbone-only weights")
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    reference = model.state_dict()
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    mismatched = {k: [list(state[k].shape) if torch.is_tensor(state[k]) else str(type(state[k])), list(reference[k].shape)] for k in set(reference) & set(state)
                  if not torch.is_tensor(state[k]) or tuple(state[k].shape) != tuple(reference[k].shape)}
    if missing or unexpected or mismatched:
        raise ValueError("Full-checkpoint mismatch; no partial load is allowed.\n" + json.dumps({
            "missing": missing, "unexpected": unexpected, "shape_mismatch": mismatched}, indent=2))
    model.load_state_dict(state, strict=True)
    report = {"path": str(checkpoint_path), "state_entries": len(state),
              "pos_embed_x": list(state["backbone.pos_embed_x"].shape),
              "pos_embed_z": list(state["backbone.pos_embed_z"].shape),
              "missing": [], "unexpected": [], "shape_mismatch": {},
              "ce_provenance": "CE has no separate parameters; strict loading does not prove training-time CE settings."}
    print("Strict full ODTrack load: {} ({} entries; no missing/unexpected keys)".format(checkpoint_path, len(state)))
    return report


def test_manifest_names():
    path = local.PROJECT_ROOT / "lib/train/data_specs/lsotb_tir120.txt"
    names = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(names) != 120 or len(set(names)) != 120:
        raise ValueError("Expected exactly 120 unique LSOTB-TIR120 names")
    return names


def checkpoint_directory():
    return Path(local.OUTPUT_DIR).expanduser().resolve() / "checkpoints/train/odtrack" / local.CONFIG_NAME


def checkpoint_for_epoch(epoch):
    return checkpoint_directory() / "ODTrack_ep{:04d}.pth.tar".format(epoch)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real training/testing. Install a CUDA-enabled PyTorch build.")


def reject_overlap(records, other_names, description):
    overlap = sorted(set(r.name for r in records) & set(other_names))
    if overlap:
        raise ValueError("{} overlap: {}".format(description, overlap))
