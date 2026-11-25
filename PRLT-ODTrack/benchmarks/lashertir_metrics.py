import csv
from pathlib import Path

import numpy as np

from .lashertir_data import rectangles

LEGACY_METRIC_PROTOCOL = "lasher_tir_aligned_init_matlab_rect_ope_s21_p51_np51_explicit_v2"
METRIC_PROTOCOL = "lasher_tir_aligned_init_matlab_colon_s21_p51_np51_v3"


def matlab_overlap_thresholds():


    thresholds = np.empty(21, dtype=np.float64)
    for k in range(10):
        thresholds[k] = k * 0.05
        thresholds[20 - k] = 1.0 - k * 0.05
    thresholds[10] = 0.5
    return thresholds


OVERLAP_THRESHOLDS = matlab_overlap_thresholds()
LEGACY_OVERLAP_THRESHOLDS = np.arange(21, dtype=np.float64) / 20.0
CENTER_THRESHOLDS = np.arange(51, dtype=np.float64)
NORMALIZED_THRESHOLDS = CENTER_THRESHOLDS / 100.0


def rect_iou(pred, gt):
    lower = np.maximum(pred[:, :2], gt[:, :2])
    upper = np.minimum(pred[:, :2] + pred[:, 2:] - 1, gt[:, :2] + gt[:, 2:] - 1)
    intersection = np.maximum(0, upper - lower + 1).prod(axis=1)
    union = pred[:, 2:].prod(axis=1) + gt[:, 2:].prod(axis=1) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return intersection / union


def sequence_errors(prediction, ground_truth):
    pred = np.asarray(prediction, dtype=np.float64).copy()
    gt = np.asarray(ground_truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4 or not len(pred):
        raise ValueError("LasHeR requires equal, nonempty N x 4 xywh predictions and GT")
    if np.isinf(pred).any() or np.isinf(gt).any():
        raise ValueError("Infinite coordinates are not supported")

    for i in range(1, len(pred)):
        if (np.isnan(pred[i]).any() or np.any(pred[i, 2:] <= 0)) and not np.isnan(gt[i]).any():
            pred[i] = pred[i - 1]
    pred[0] = gt[0]
    center = pred[:, :2] + (pred[:, 2:] - 1) / 2
    gt_center = gt[:, :2] + (gt[:, 2:] - 1) / 2
    err = np.sqrt(((center - gt_center) ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.sqrt(((center / gt[:, 2:] - gt_center / gt[:, 2:]) ** 2).sum(axis=1))
    valid = np.all(gt > 0, axis=1)
    overlap = np.full(len(gt), -1.0)
    overlap[valid] = rect_iou(pred[valid], gt[valid])


    err[~valid] = -1.0
    normalized[~valid] = -1.0
    return overlap, err, normalized


def sequence_curves(prediction, ground_truth):
    overlap, center, normalized = sequence_errors(prediction, ground_truth)
    return curves_from_errors(overlap, center, normalized)


def curves_from_errors(overlap, center, normalized):
    return ((overlap[:, None] > OVERLAP_THRESHOLDS).mean(axis=0),
            (center[:, None] <= CENTER_THRESHOLDS).mean(axis=0),
            (normalized[:, None] <= NORMALIZED_THRESHOLDS).mean(axis=0))


def sequence_curves_with_audit(prediction, ground_truth):

    overlap, center, normalized = sequence_errors(prediction, ground_truth)
    curves = curves_from_errors(overlap, center, normalized)
    legacy_decisions = overlap[:, None] > LEGACY_OVERLAP_THRESHOLDS
    corrected_decisions = overlap[:, None] > OVERLAP_THRESHOLDS
    changed = legacy_decisions != corrected_decisions
    audit = dict(affected_frames=int(changed.any(axis=1).sum()),
                 changed_threshold_decisions=int(changed.sum()),
                 changed_decisions_by_threshold=changed.sum(axis=0).tolist(),
                 invalid_GT_rows=int((~np.all(np.asarray(ground_truth) > 0, axis=1)).sum()))
    return curves, legacy_decisions.mean(axis=0), audit


def score_curves(success, precision, normalized):
    return {"Success_pct": float(np.mean(success) * 100),
            "Precision_20px_pct": float(precision[20] * 100),
            "NormPrec_AUC_pct": float(np.mean(normalized) * 100),
            "NormPrec_0.20_pct": float(normalized[20] * 100)}


def aggregate(success, precision, normalized):
    curves, excluded = [], {}
    for name, values in (("success", success), ("precision", precision), ("norm_precision", normalized)):
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError("Cannot aggregate empty sequence curves")
        keep = matrix.sum(axis=1) > np.finfo(np.float64).eps
        if not keep.any():
            raise ValueError("No nonzero sequence curves for " + name)
        curves.append(matrix[keep].mean(axis=0))
        excluded[name] = int((~keep).sum())
    return score_curves(*curves), curves, excluded


def compute_metrics(data, prediction_directory, report_directory, write_json):
    rows, success, precision, normalized = [], [], [], []
    prediction_directory, report_directory = Path(prediction_directory), Path(report_directory)

    predictions = [rectangles(prediction_directory / (r.name + ".txt"), len(r.boxes)) for r in data.records]
    for record, pred in zip(data.records, predictions):
        s, p, n = sequence_curves(pred, record.boxes)
        success.append(s); precision.append(p); normalized.append(n)
        rows.append(dict(sequence=record.name, frames=len(record.boxes), annotation_file=str(record.annotation_path),
                         **score_curves(s, p, n)))
    result, curves, excluded = aggregate(success, precision, normalized)
    result.update(benchmark="LasHeR-TIR", split="official_test_245" if data.complete_official_split else "regression_fixture",
                  sequences=len(rows), frames=sum(len(r.boxes) for r in data.records),
                  complete_official_split=data.complete_official_split,
                  input_modality="TIR only", ground_truth_modality="aligned shared GT",
                  annotation_policy=data.annotation_policy,
                  metric_protocol=METRIC_PROTOCOL, annotation_identity=data.annotation_identity,
                  zero_sequence_curves_excluded=excluded,
                  prediction_directory=str(prediction_directory),
                  metric_definition={"Success_pct": "Equal-sequence mean of 21 IoU>threshold rates at 0:0.05:1",
                      "Precision_20px_pct": "Equal-sequence center precision at <=20 pixels",
                      "NormPrec_AUC_pct": "Equal-sequence mean of 51 normalized precision rates at 0:0.01:0.5",
                      "NormPrec_0.20_pct": "Equal-sequence normalized precision at <=0.20; distinct from NP-AUC",
                      "GT_policy": "MATLAB rectangle OPE: all four GT values >0; other errors -1, no absent-frame exclusion",
                      "weighting": "Sequence curves are averaged equally; never frame-weighted"})
    report_directory.mkdir(parents=True, exist_ok=True)
    with (report_directory / "per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (report_directory / "curves.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["curve", "threshold", "rate", "percent"])
        for name, thresholds, rates in zip(("success", "precision", "norm_precision"),
                (OVERLAP_THRESHOLDS, CENTER_THRESHOLDS, NORMALIZED_THRESHOLDS), curves):
            w.writerows((name, float(t), float(v), float(v * 100)) for t, v in zip(thresholds, rates))
    with (report_directory / "curves.npz.tmp").open("wb") as f:
        np.savez_compressed(f, success=curves[0], precision=curves[1], norm_precision=curves[2],
                            overlap_thresholds=OVERLAP_THRESHOLDS, center_thresholds=CENTER_THRESHOLDS,
                            normalized_thresholds=NORMALIZED_THRESHOLDS,
                            per_sequence_success=np.asarray(success), per_sequence_precision=np.asarray(precision),
                            per_sequence_norm_precision=np.asarray(normalized), sequence_names=np.asarray(data.names))
    (report_directory / "curves.npz.tmp").replace(report_directory / "curves.npz")
    write_json(report_directory / "metrics.json", result)
    return result
