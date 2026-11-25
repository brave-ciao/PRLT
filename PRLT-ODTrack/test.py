import argparse
from pathlib import Path
from types import SimpleNamespace

import local_settings as local
from benchmarks.datasets import BENCHMARKS
from prlt.io import checkpoint_for_epoch, load_config


def main():
    names = [key for key, _, _ in BENCHMARKS]
    parser = argparse.ArgumentParser(description="Evaluate PRLT-ODTrack on the TIR test benchmarks.")
    parser.add_argument("--checkpoint", type=Path, default=local.TEST_CHECKPOINT or None)
    parser.add_argument("--dataset", nargs="+", choices=names, default=names)
    parser.add_argument("--output", type=Path, default=local.TEST_OUTPUT_DIR)
    parser.add_argument("--device", default=f"cuda:{local.GPU_ID}")
    args = parser.parse_args()

    import torch
    from benchmarks.runner import run

    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        parser.error("--device must be cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            parser.error("--device must select an available CUDA device")
        torch.cuda.set_device(device.index if device.index is not None else 0)
    cfg = load_config()
    checkpoint = (args.checkpoint or checkpoint_for_epoch(cfg.TEST.EPOCH)).expanduser().resolve()
    options = SimpleNamespace(
        lsotb_root=local.LSOTB120_ROOT, lsotb_annotation_root=local.LSOTB120_ANNOTATION_ROOT,
        ptb_root=local.PTBTIR_ROOT, ptb_annotation_root=local.PTBTIR_ANNOTATION_ROOT,
        vtuav_root=local.VTUAVTIR_ROOT, vtuav_st_root=local.VTUAVTIR_ST_ROOT,
        vtuav_image_subdir=local.VTUAVTIR_IMAGE_SUBDIR,
        lasher_root=local.LASHERTIR_ROOT, lasher_annotation_root=local.LASHERTIR_ANNOTATION_ROOT,
        lasher_image_subdir=local.LASHERTIR_IMAGE_SUBDIR,
    )
    run(checkpoint, args.output.expanduser().resolve(), args.dataset, options, device, cfg)


if __name__ == "__main__":
    main()
