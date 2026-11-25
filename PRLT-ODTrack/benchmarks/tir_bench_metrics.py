from pathlib import Path
import csv

import numpy as np

from .tir_bench_data import rectangles

PROTOCOLS = {
    "ptbtir": "ptbtir_official_ope_21iou_51center_v1",
    "vtuavtir": "vtuav_ir_only_official_stride10_21iou_51center_v1",
}
OVERLAP_THRESHOLDS = np.arange(21, dtype=np.float64) / 20.0
CENTER_THRESHOLDS = np.arange(51, dtype=np.float64)


def prediction_file(directory, benchmark, name):
    return Path(directory) / benchmark / (name + ".txt")


def validated_prediction(path, frame_count):
    values = rectangles(path)
    if values.shape != (frame_count, 4) or np.any(values[:, 2:] < 0):
        raise ValueError("Found shape {}; expected ({}, 4) finite xywh predictions with nonnegative sizes, "
                         "one per FULL-RATE frame: {}".format(values.shape, frame_count, path))
    return values


def rect_iou(a, b):

    left = np.maximum(a[:, :2], b[:, :2])
    right = np.minimum(a[:, :2] + a[:, 2:] - 1, b[:, :2] + b[:, 2:] - 1)
    intersection = np.maximum(0.0, right - left + 1).prod(axis=1)
    union = a[:, 2:].prod(axis=1) + b[:, 2:].prod(axis=1) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return intersection / union


def sequence_errors(benchmark, prediction, ground_truth):
    pred = np.asarray(prediction, dtype=np.float64).copy()
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4 or not len(pred):
        raise ValueError("Scored predictions and GT must have equal nonempty N x 4 shapes")
    if benchmark == "ptbtir":


        for i in range(1, len(pred)):
            if (np.isnan(pred[i]).any() or np.any(pred[i, 2:] <= 0)) and not np.isnan(gt[i]).any():
                pred[i] = pred[i - 1]
        pred[0] = gt[0]
        centers = pred[:, :2] + (pred[:, 2:] - 1) / 2
        gt_centers = gt[:, :2] + (gt[:, 2:] - 1) / 2
        valid = np.all(gt > 0, axis=1)
        overlap = np.full(len(gt), -1.0)
        overlap[valid] = rect_iou(pred[valid], gt[valid])
        error = np.sqrt(((centers - gt_centers) ** 2).sum(axis=1))
        error[~valid] = -1.0
    elif benchmark == "vtuavtir":


        overlap = rect_iou(pred, gt)
        centers = pred[:, :2] + pred[:, 2:] / 2
        gt_centers = gt[:, :2] + gt[:, 2:] / 2
        error = np.sqrt(((centers - gt_centers) ** 2).sum(axis=1))


    else:
        raise ValueError("Unknown benchmark: " + str(benchmark))
    return overlap, error


def sequence_curves(benchmark, prediction, ground_truth):
    overlap, center = sequence_errors(benchmark, prediction, ground_truth)
    success = (overlap[:, None] > OVERLAP_THRESHOLDS[None, :]).mean(axis=0)
    precision = (center[:, None] <= CENTER_THRESHOLDS[None, :]).mean(axis=0)
    return success, precision


def aggregate(benchmark, rows, success, precision):
    if not rows:
        raise ValueError("Cannot score an empty subset")
    s, p = np.asarray(success), np.asarray(precision)


    s_keep = s.sum(axis=1) > np.finfo(np.float64).eps if benchmark == "ptbtir" else np.ones(len(rows), dtype=bool)
    p_keep = p.sum(axis=1) > np.finfo(np.float64).eps if benchmark == "ptbtir" else np.ones(len(rows), dtype=bool)
    sc = s[s_keep].mean(axis=0) if s_keep.any() else np.zeros(21)
    pc = p[p_keep].mean(axis=0) if p_keep.any() else np.zeros(51)
    result = {"sequences": len(rows), "tracking_frames": sum(r["tracking_frames"] for r in rows),
              "scored_frames": sum(r["scored_frames"] for r in rows),
              "Success_pct": float(sc.mean() * 100), "Precision_20px_pct": float(pc[20] * 100),
              "zero_success_curves_excluded": int((~s_keep).sum()),
              "zero_precision_curves_excluded": int((~p_keep).sum())}
    if benchmark == "vtuavtir":
        result.update(IR_Success_pct=result["Success_pct"], IR_Precision_20px_pct=result["Precision_20px_pct"])
    return result, sc, pc


def compute_metrics(data, directory, write_json, prediction_directory=None):
    rows, success, precision = [], [], []
    for record in data.records:
        path = (Path(prediction_directory) / (record.name + ".txt") if prediction_directory is not None
                else prediction_file(directory, data.benchmark, record.name))
        full = validated_prediction(path, len(record.frames))

        selected = full[record.score_indices]
        s, p = sequence_curves(data.benchmark, selected, record.boxes)
        success.append(s)
        precision.append(p)
        rows.append({"sequence": record.name, "split": record.split,
                     "tracking_frames": len(record.frames), "scored_frames": len(record.boxes),
                     "Success_pct": float(s.mean() * 100), "Precision_20px_pct": float(p[20] * 100)})
    result, sc, pc = aggregate(data.benchmark, rows, success, precision)
    result.update(benchmark=data.protocol["benchmark"], split=data.split,
                  metric_protocol=PROTOCOLS[data.benchmark],
                  metric_definition="Equal sequence weighting; Success=arithmetic mean of 21 rates IoU>0:0.05:1; Precision=rate center error<=20px; integer ODTrack exported boxes.",
                  annotation_format="xywh", prediction_format="xywh",
                  score_stride=10 if data.benchmark == "vtuavtir" else 1)
    arrays = {"overlap_thresholds": OVERLAP_THRESHOLDS, "center_thresholds": CENTER_THRESHOLDS,
              "success": sc, "precision": pc, "per_sequence_success": np.asarray(success),
              "per_sequence_precision": np.asarray(precision), "sequence_names": np.asarray(data.names)}
    if data.benchmark == "vtuavtir":
        result["metric_scope"] = "IR only. RGB annotations are neither read nor used. Not the dual-GT maximum MSR/MPR protocol."
        result["groups"] = {}
        for split in ("st", "lt"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split]
            if idx:
                group, gs, gp = aggregate(data.benchmark, [rows[i] for i in idx],
                                           [success[i] for i in idx], [precision[i] for i in idx])
                result["groups"][split.upper()] = group
                arrays[split + "_success"], arrays[split + "_precision"] = gs, gp
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (directory / "curves.npz.tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    (directory / "curves.npz.tmp").replace(directory / "curves.npz")
    write_json(directory / "metrics.json", result)
    return result
