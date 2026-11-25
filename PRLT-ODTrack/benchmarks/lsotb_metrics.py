from pathlib import Path

import numpy as np

from .tir_bench_metrics import rect_iou, validated_prediction


PROTOCOL = "lsotb_matlab_norm_precision_auc_51_v2"
CENTER_THRESHOLDS = np.arange(51, dtype=np.float64)
NORMALIZED_THRESHOLDS = CENTER_THRESHOLDS / 100.0


def sequence_curves(prediction, ground_truth, overlap_thresholds):
    prediction = np.asarray(prediction, dtype=np.float64).copy()
    gt = np.asarray(ground_truth, dtype=np.float64)
    for i in np.flatnonzero(np.any(prediction[1:, 2:] == 0, axis=1)) + 1:
        prediction[i] = prediction[i - 1]
    prediction[0] = gt[0]
    pred_center = prediction[:, :2] + 0.5 * (prediction[:, 2:] - 1.0)
    gt_center = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    valid = (gt[:, 2:] > 0).all(axis=1)
    overlap = np.full(len(gt), -1.0)
    overlap[valid] = rect_iou(prediction[valid], gt[valid])
    error = np.full(len(gt), np.inf)
    error[valid] = np.sqrt(((pred_center[valid] - gt_center[valid]) ** 2).sum(axis=1))


    valid_norm = (gt > 0).all(axis=1)
    normalized = np.full(len(gt), -1.0)
    normalized[valid_norm] = np.sqrt(((pred_center[valid_norm] / gt[valid_norm, 2:]
                                      - gt_center[valid_norm] / gt[valid_norm, 2:]) ** 2).sum(axis=1))
    return ((overlap[:, None] > overlap_thresholds).sum(axis=0).astype(np.float32) / len(gt),
            (error[:, None] <= CENTER_THRESHOLDS).sum(axis=0).astype(np.float32) / len(gt),
            (normalized[:, None] <= NORMALIZED_THRESHOLDS).mean(axis=0))


def summary_values(success, precision, normalized):
    return {
        "Success_pct": float(success.mean() * 100.0),
        "Precision_20px_pct": float(precision[20] * 100.0),
        "NormPrec_AUC_pct": float(normalized.mean() * 100.0),
        "NormPrec_0.20_pct": float(normalized[20] * 100.0),
    }


def compute_metrics(records, prediction_dir, report_dir, write_csv, write_json):
    import torch


    overlap_thresholds = torch.arange(0.0, 1.0 + 0.05, 0.05, dtype=torch.float64).numpy()
    rows, curves = [], []
    for record in records:
        prediction = validated_prediction(prediction_dir / (record.name + ".txt"), len(record.frames))
        sequence = sequence_curves(prediction, record.boxes, overlap_thresholds)
        curves.append(sequence)
        rows.append({"sequence": record.name, "frames": len(record.frames), **summary_values(*sequence)})
    success, precision, normalized = [np.stack(parts, axis=0) for parts in zip(*curves)]
    mean_success, mean_precision, mean_normalized = [c.mean(axis=0) for c in (success, precision, normalized)]
    metrics = dict(benchmark="LSOTB-TIR120", sequences=len(rows), frames=sum(r["frames"] for r in rows),
                   metric_protocol=PROTOCOL, **summary_values(mean_success, mean_precision, mean_normalized))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "per_sequence.csv", rows)
    np.savez_compressed(report_dir / "curves.npz",
                        success=mean_success, precision=mean_precision, norm_precision=mean_normalized,
                        overlap_thresholds=overlap_thresholds, center_thresholds=CENTER_THRESHOLDS,
                        normalized_thresholds=NORMALIZED_THRESHOLDS,
                        per_sequence_success=success, per_sequence_precision=precision,
                        per_sequence_norm_precision=normalized, sequence_names=np.asarray([r.name for r in records]))
    write_json(report_dir / "metrics.json", metrics)
    return metrics
