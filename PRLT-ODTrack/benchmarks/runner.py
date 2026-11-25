import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from prlt.io import file_sha256, json_digest, write_json
from . import lashertir_metrics, lsotb_metrics, tir_bench_data, tir_bench_metrics, vtuav_msr_mpr
from .datasets import BENCHMARKS, load_benchmark


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_prediction(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            np.savetxt(stream, boxes, fmt="%d", delimiter="\t")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def track_sequence(tracker, record):
    prediction = np.empty((len(record.frames), 4), dtype=np.float64)
    prediction[0] = record.boxes[0]
    tracker.initialize(tir_bench_data.read_ir(record.frames[0]), {"init_bbox": prediction[0].tolist()})
    for index, frame in enumerate(record.frames[1:], start=1):
        output = tracker.track(tir_bench_data.read_ir(frame))
        box = np.asarray(output["target_bbox"], dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError(f"{record.name}, frame {index + 1}: invalid xywh prediction {box}")
        prediction[index] = box
    return prediction.astype(np.int64)


def score(benchmark, prediction_dir, report_dir):
    if benchmark.key == "lsotb120":
        return lsotb_metrics.compute_metrics(benchmark.records, prediction_dir, report_dir, write_csv, write_json)
    if benchmark.key == "ptbtir":
        return tir_bench_metrics.compute_metrics(benchmark.data, report_dir, write_json,
                                                prediction_directory=prediction_dir)
    if benchmark.key == "vtuavtir":
        meta, annotations = benchmark.vtuav_gt
        return vtuav_msr_mpr.evaluate_directory(prediction_dir, meta, annotations, report_dir)
    return lashertir_metrics.compute_metrics(benchmark.data, prediction_dir, report_dir, write_json)


def summary_row(benchmark, metrics):
    row = dict(benchmark=benchmark.label, sequences=len(benchmark.records), SR_pct="", PR_pct="",
               NPR_pct="", MSR_pct="", MPR_pct="")
    if benchmark.key == "vtuavtir":
        row.update(MSR_pct=metrics["MSR_pct"], MPR_pct=metrics["MPR_20px_pct"])
    else:
        row.update(SR_pct=metrics["Success_pct"], PR_pct=metrics["Precision_20px_pct"])
    if benchmark.key == "lsotb120":
        row["NPR_pct"] = metrics["NormPrec_AUC_pct"]
    elif benchmark.key == "lashertir":
        row["NPR_pct"] = metrics["NormPrec_0.20_pct"]
    return row


def run(checkpoint, output, datasets, options, device, cfg):
    import cv2
    import torch
    from lib.test.tracker.odtrack import ODTrack
    from lib.test.utils.params import TrackerParams

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Tracking checkpoint not found: {checkpoint}")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(checkpoint)
    params.device = device
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.debug = 0
    params.save_all_boxes = False
    tracker = None
    checkpoint_hash = file_sha256(checkpoint)
    out = output / f"{checkpoint.name.split('.')[0]}_{checkpoint_hash[:12]}"
    rows = []
    for key, label, count in BENCHMARKS:
        if key not in datasets:
            continue
        print(f"Reading {label} ({count} sequences)", flush=True)
        benchmark = load_benchmark(key, label, count, options)
        identity = json_digest({"data": benchmark.identity, "config": cfg.TEST})
        dataset_output = out / key / identity[:12]
        prediction_dir = dataset_output / "predictions"
        for index, record in enumerate(benchmark.records, start=1):
            path = prediction_dir / f"{record.name}.txt"
            if path.is_file():
                try:
                    tir_bench_metrics.validated_prediction(path, len(record.frames))
                except ValueError:
                    pass
                else:
                    continue
            if tracker is None:
                tracker = ODTrack(params)
            start = perf_counter()
            with torch.no_grad():
                prediction = track_sequence(tracker, record)
            write_prediction(path, prediction)
            print(f"{label} [{index}/{count}] {record.name}: {len(record.frames)} frames in "
                  f"{perf_counter() - start:.1f} s", flush=True)
        metrics = score(benchmark, prediction_dir, dataset_output / "metrics")
        row = summary_row(benchmark, metrics)
        rows.append(row)
        display = "  ".join(f"{name.removesuffix('_pct')}={score:.2f}%"
                            for name, score in row.items() if name.endswith("_pct") and score != "")
        print(f"{label}  {display}", flush=True)
    write_csv(out / "summary.csv", rows)
    print(f"Summary: {out / 'summary.csv'}", flush=True)
    return out
