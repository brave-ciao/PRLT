from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import lashertir_data, tir_bench_data, vtuav_msr_mpr

BENCHMARKS = (
    ("lsotb120", "LSOTB-TIR120", 120),
    ("ptbtir", "PTB-TIR", 60),
    ("vtuavtir", "VTUAV-TIR ST", 176),
    ("lashertir", "LasHeR-TIR", 245),
)


@dataclass
class Benchmark:
    key: str
    label: str
    data: object
    identity: str
    vtuav_gt: tuple | None = None

    @property
    def records(self):
        return self.data.records


@dataclass
class LSOTBRecord:
    name: str
    frames: list[Path]
    boxes: np.ndarray


def load_lsotb(root, annotation_root):
    manifest = Path(__file__).with_name("lsotb_tir120_protocol.json")
    reference = json.loads(manifest.read_text(encoding="utf-8"))["sequence_records"]
    if len(reference) != 120 or sum(s["frame_count"] for s in reference) != 82133:
        raise ValueError("The bundled LSOTB-TIR120 manifest is inconsistent.")

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Set LSOTB120_ROOT in local_settings.py. Missing: {root}")

    candidates = (root, root / "LSOTB-TIR", root / "LSOTB-TIR120", root / "LSOTB-TIR120" / "LSOTB-TIR")
    complete = list(dict.fromkeys(p.resolve() for p in candidates
                                   if all((p / spec["name"]).is_dir() for spec in reference)))
    if len(complete) != 1:
        raise ValueError(f"Set LSOTB120_ROOT to one complete 120-sequence directory. Matches: {complete or root}")
    root = complete[0]
    override = Path(annotation_root).expanduser().resolve() if annotation_root else None

    records = []
    fingerprint = hashlib.sha256(str(root).encode())
    for spec in reference:
        name = spec["name"]
        base = root / name
        candidates = [p for p in (base, base / "img", base / "imgs") if p.is_dir()
                      and any(f.is_file() and f.suffix.lower() in tir_bench_data.IMAGE_EXTENSIONS
                              for f in p.iterdir())]
        if len(candidates) != 1:
            raise ValueError(f"{name}: expected one LSOTB image directory: sequence/, img/, or imgs/.")
        frames, ids = tir_bench_data.numeric_images(candidates[0])
        expected = np.arange(spec["startFrame"], spec["endFrame"] + 1)
        if not np.array_equal(ids, expected):
            raise ValueError(f"{name}: frame IDs differ from the official LSOTB-TIR120 test split.")

        if override:
            annotation = override / (name + ".txt")
        else:
            paths = [p for p in (base / "groundtruth.txt", base / "groundtruth_rect.txt") if p.is_file()]
            if not paths:
                raise FileNotFoundError(f"{name}: groundtruth.txt / groundtruth_rect.txt missing in {base}.")
            annotation = paths[0]
            if len(paths) == 2:
                other = tir_bench_data.rectangles(paths[1])
                if tir_bench_data.numeric_digest(other) != spec["annotation_xywh_sha256"]:
                    raise ValueError(f"{name}: conflicting LSOTB annotation files in {base}.")

        boxes = tir_bench_data.rectangles(annotation)
        if (len(boxes) != len(frames) or tir_bench_data.numeric_digest(boxes) != spec["annotation_xywh_sha256"]
                or np.any(boxes[0, 2:] <= 0)):
            raise ValueError(f"{name}: GT does not match the official LSOTB-TIR120 xywh annotation: {annotation}")
        records.append(LSOTBRecord(name, frames, boxes))
        fingerprint.update((name + spec["annotation_xywh_sha256"]).encode())
        for path in frames:
            stat = path.stat()
            fingerprint.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())

    return SimpleNamespace(records=records, identity=fingerprint.hexdigest())


def load_benchmark(key, label, count, options):
    if key == "lsotb120":
        data = load_lsotb(options.lsotb_root, options.lsotb_annotation_root)
        benchmark = Benchmark(key, label, data, data.identity)
    elif key == "ptbtir":
        data = tir_bench_data.BenchmarkData(key, SimpleNamespace(
            DATA_ROOT=options.ptb_root, ANNOTATION_ROOT=options.ptb_annotation_root))
        benchmark = Benchmark(key, label, data, data.identity)
    elif key == "vtuavtir":
        data = tir_bench_data.BenchmarkData(key, SimpleNamespace(
            DATA_ROOT=options.vtuav_root, ST_ROOT=options.vtuav_st_root,
            IMAGE_SUBDIR=options.vtuav_image_subdir, SPLIT="st"))
        meta = vtuav_msr_mpr.load_reference()
        frame_counts = {r["name"]: r["frame_count"] for r in meta["records"]}
        for record in data.records:
            if len(record.frames) != frame_counts[record.name]:
                raise ValueError(f"{record.name}: full-rate VTUAV frame count differs from the ST176 reference.")
        annotations = vtuav_msr_mpr.load_annotations(meta, data.records)
        benchmark = Benchmark(key, label, data, data.identity, (meta, annotations))
    elif key == "lashertir":
        data = lashertir_data.LasHeRTIRData(SimpleNamespace(
            DATA_ROOT=options.lasher_root, ANNOTATION_ROOT=options.lasher_annotation_root,
            ANNOTATION_FILENAME="init.txt", IMAGE_SUBDIR=options.lasher_image_subdir))
        identity = lashertir_data.digest([str(data.root), data.annotation_identity, data.image_identity])
        benchmark = Benchmark(key, label, data, identity)
    else:
        raise ValueError(f"Unsupported benchmark {key!r}.")

    if len(benchmark.records) != count or len({r.name for r in benchmark.records}) != count:
        raise ValueError(f"{label}: the complete official {count}-sequence test split is required.")
    return benchmark
