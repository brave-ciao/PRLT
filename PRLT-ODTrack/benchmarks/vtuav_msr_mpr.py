import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .tir_bench_data import rectangles, numeric_digest
from .tir_bench_metrics import rect_iou, validated_prediction, OVERLAP_THRESHOLDS

REFERENCE = Path(__file__).resolve().parent / "vtuav_msr_reference"
PROTOCOL = "VTUAV_OFFICIAL_DUAL_GT_MAXIMUM_ST176_v1"
CENTER_THRESHOLDS = np.arange(51, dtype=np.float64)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_reference():
    meta = json.loads((REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    if (meta.get("sequence_count") != 176 or len(meta["records"]) != 176
            or len({r["name"] for r in meta["records"]}) != 176 or meta.get("annotation_stride") != 10):
        raise ValueError("Reference must describe the complete official ST176 split and stride 10")
    return meta


def load_annotations(meta, records):
    annotations = {}
    ir_files = {r.name: r.annotation_path for r in records}
    for record in meta["records"]:
        name = record["name"]
        ir_path = ir_files[name]
        rgb_path = ir_path.with_name("rgb.txt")
        ir, rgb = rectangles(ir_path), rectangles(rgb_path)
        if (ir.shape != (record["annotation_count"], 4) or rgb.shape != ir.shape
                or numeric_digest(ir) != record["IR_GT_numeric_sha256"]
                or numeric_digest(rgb) != record["RGB_GT_numeric_sha256"]):
            raise ValueError(f"{name}: ir.txt and rgb.txt must contain the official xywh annotations.")
        annotations[name] = (ir, rgb)
    return annotations


def choose_prediction_folder(directory, names):
    directory = Path(directory).expanduser().resolve()
    choices = [directory, directory/"vtuavtir"]
    complete = [p for p in choices if all((p/(name+".txt")).is_file() for name in names)]
    if not complete:
        counts = [(str(p), sum((p/(name+".txt")).is_file() for name in names)) for p in choices]
        raise FileNotFoundError("No complete ST176 prediction layout in {}. Found {}. "
                                "Flat and nested files are not combined.".format(directory, counts))
    if len(complete) == 2:
        for name in names:
            if not np.array_equal(rectangles(complete[0]/(name+".txt")), rectangles(complete[1]/(name+".txt"))):
                raise ValueError("Two complete layouts contain different predictions: " + str(directory))
    return complete[0]


def protocol_curves(prediction, ir_gt, rgb_gt):
    p, ir, rgb = [np.asarray(x, dtype=np.float64) for x in (prediction, ir_gt, rgb_gt)]
    if p.shape != ir.shape or p.shape != rgb.shape or p.ndim != 2 or p.shape[1] != 4 or not len(p):
        raise ValueError("Predictions and both GT arrays must have equal nonempty N x 4 shapes")
    ir_overlap, rgb_overlap = rect_iou(p, ir), rect_iou(p, rgb)
    center = p[:, :2] + p[:, 2:] / 2.0
    ir_center = ir[:, :2] + ir[:, 2:] / 2.0
    rgb_center = rgb[:, :2] + rgb[:, 2:] / 2.0
    ir_error = np.sqrt(np.sum((center - ir_center)**2, axis=1))
    rgb_error = np.sqrt(np.sum((center - rgb_center)**2, axis=1))


    overlap = np.maximum(ir_overlap, rgb_overlap)
    error = np.minimum(ir_error, rgb_error)
    return {
        "IR_success": (ir_overlap[:, None] > OVERLAP_THRESHOLDS[None, :]).mean(axis=0),
        "IR_precision": (ir_error[:, None] <= CENTER_THRESHOLDS[None, :]).mean(axis=0),
        "MSR": (overlap[:, None] > OVERLAP_THRESHOLDS[None, :]).mean(axis=0),
        "MPR": (error[:, None] <= CENTER_THRESHOLDS[None, :]).mean(axis=0),
    }


def evaluate_directory(directory, meta, annotations, output_directory):
    names = [r["name"] for r in meta["records"]]
    folder = choose_prediction_folder(directory, names)
    curves, rows = [], []
    for record in meta["records"]:
        name = record["name"]
        path = folder / (name + ".txt")
        full = validated_prediction(path, record["frame_count"])
        selected = full[::10]
        ir_gt, rgb_gt = annotations[name]
        curve = protocol_curves(selected, ir_gt, rgb_gt)
        digest = sha256(path)
        curves.append(curve)
        rows.append({"sequence": name, "tracking_frames": len(full), "scored_frames": len(selected),
                     "IR_Success_pct": float(curve["IR_success"].mean()*100),
                     "IR_Precision_20px_pct": float(curve["IR_precision"][20]*100),
                     "MSR_pct": float(curve["MSR"].mean()*100), "MPR_20px_pct": float(curve["MPR"][20]*100),
                     "prediction_sha256": digest})
    avg = {key: np.mean([c[key] for c in curves], axis=0) for key in curves[0]}
    result = {"benchmark": "VTUAV-ST", "sequences": len(rows), "metric_protocol": PROTOCOL,
              "source_prediction_directory": str(folder),
              "tracking_frames": sum(r["tracking_frames"] for r in rows),
              "scored_frames": sum(r["scored_frames"] for r in rows),
              "IR_Success_pct": float(avg["IR_success"].mean()*100),
              "IR_Precision_20px_pct": float(avg["IR_precision"][20]*100),
              "MSR_pct": float(avg["MSR"].mean()*100), "MPR_20px_pct": float(avg["MPR"][20]*100),
              "input_policy": "Full-rate infrared predictions.",
              "annotation_sources": "Official ir.txt and rgb.txt per sequence."}
    result["MSR_minus_IR_Success_pp"] = result["MSR_pct"]-result["IR_Success_pct"]
    result["MPR_minus_IR_Precision_pp"] = result["MPR_20px_pct"]-result["IR_Precision_20px_pct"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory/"per_sequence.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_directory/"curves.npz", **avg, overlap_thresholds=OVERLAP_THRESHOLDS,
                        center_thresholds=CENTER_THRESHOLDS)
    write_json(output_directory/"summary.json", result)
    return result
