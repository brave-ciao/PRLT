from pathlib import Path
import random
import numpy as np
import torch

from lib.train.trainers import LTRTrainer
from .io import trusted_torch_load
from .ptr import PREFIXES, PTR_PREFIX, TARGET_PREFIX, tracking_state_dict, auxiliary_state_dict
from .options import protocol


def capture_rng_state(device):
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": cuda_state}


def restore_rng_state(state, device):
    if (state["torch_cuda"] is not None) != (device.type == "cuda"):
        raise ValueError("Restore the training checkpoint on the same device type.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


class PRLTTrainer(LTRTrainer):
    def save_checkpoint(self):
        net = getattr(self.actor.net, "module", self.actor.net)
        state = {"epoch": self.epoch, "actor_type": type(self.actor).__name__, "net_type": "ODTrack",
                 "net": tracking_state_dict(net), "prlt_aux": auxiliary_state_dict(net),
                 "prlt_protocol": protocol(), "optimizer": self.optimizer.state_dict(),
                 "lr_scheduler": self.lr_scheduler.state_dict(), "stats": self.stats, "settings": self.settings,
                 "actor_step": int(getattr(self.actor, "_step", 0)), "rng_state": capture_rng_state(self.device),
                 "net_info": getattr(net, "info", None), "constructor": getattr(net, "constructor", None)}
        if any(k.startswith(PREFIXES) for k in state["net"]):
            raise RuntimeError("PRLT parameters leaked into vanilla ODTrack weights")
        directory = Path(self._checkpoint_dir)/self.settings.project_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/"ODTrack_ep{:04d}.pth.tar".format(self.epoch)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        if fields is not None or ignore_fields is not None:
            raise ValueError("PRLT resume must restore the complete training state")
        directory = Path(self._checkpoint_dir)/self.settings.project_path
        if checkpoint is None:
            candidates = sorted(directory.glob("ODTrack_ep*.pth.tar"))
            if not candidates:
                raise FileNotFoundError("No complete PRLT checkpoint to resume under " + str(directory))
            path = candidates[-1]
        elif isinstance(checkpoint, int):
            path = directory/"ODTrack_ep{:04d}.pth.tar".format(checkpoint)
        else:
            path = Path(checkpoint).expanduser()
            if path.is_dir():
                candidates = sorted(path.glob("ODTrack_ep*.pth.tar"))
                if not candidates:
                    raise FileNotFoundError(path)
                path = candidates[-1]
        state = trusted_torch_load(path)
        required = ("net", "prlt_aux", "prlt_protocol", "optimizer", "lr_scheduler", "epoch", "stats", "actor_step", "rng_state")
        if any(key not in state for key in required) or state["prlt_protocol"] != protocol():
            raise ValueError("Checkpoint is not this PRLT experiment, or PRLT settings differ: " + str(path))
        net = getattr(self.actor.net, "module", self.actor.net)
        combined = dict(state["net"])
        if any(key.startswith(PREFIXES) for key in combined):
            raise ValueError("Expected a pure ODTrack 'net' state")
        if state["prlt_aux"]:
            combined.update({PTR_PREFIX+key: value for key, value in state["prlt_aux"]["ptr"].items()})
            combined.update({TARGET_PREFIX+key: value for key, value in state["prlt_aux"]["target_refiner"].items()})
        net.load_state_dict(combined, strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.epoch, self.stats = int(state["epoch"]), state["stats"]
        self.actor._step = int(state["actor_step"])
        net.track_query = None

        restore_rng_state(state["rng_state"], self.device)
        print("Resumed ODTrack + PRLT, optimizer, scheduler, teacher step and RNG from:", path, flush=True)
        return True
