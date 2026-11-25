from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def sequence_names(root, list_file=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("LSOTB root does not exist: {}. Edit local_settings.py".format(root))
    listing = Path(list_file) if list_file else root / "list.txt"
    if listing.is_file():
        names = [s.strip() for s in listing.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    else:
        names = sorted([p.name for p in root.iterdir() if p.is_dir() and
                        ((p / "groundtruth.txt").is_file() or (p / "groundtruth_rect.txt").is_file())],
                       key=natural_key)
    if not names:
        raise ValueError("No sequences found under {}".format(root))
    if len(names) != len(set(names)):
        raise ValueError("Duplicate sequence names in {}".format(listing))
    for name in names:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts or len(parts) != 1 or "/" in name or "\\" in name:
            raise ValueError("Expected one sequence directory name, got {!r}".format(name))
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
    return names


def read_boxes(path):

    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = re.split(r"[,;\s]+", line.strip())
        if len(fields) != 4:
            raise ValueError("{}:{} expected four box columns, found {} fields".format(path, line_no, len(fields)))
        rows.append([float(x) for x in fields])
    result = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Empty or non-finite annotation: {}".format(path))
    return result


def boxes_to_xywh(raw_boxes, box_format="xywh"):


    box_format = str(box_format).strip().lower()
    if box_format not in ("xywh", "xyxy"):
        raise ValueError("Box format must be explicitly 'xywh' or 'xyxy', got {!r}".format(box_format))
    result = np.asarray(raw_boxes, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("Expected an N x 4 annotation array")
    if box_format == "xyxy":
        result[:, 2:] -= result[:, :2]
    return result


@dataclass
class SequenceFiles:
    name: str
    directory: Path
    frames: list
    boxes: np.ndarray
    annotation_path: Path
    raw_boxes: np.ndarray = None
    box_format: str = "xywh"


def read_sequence(root, name, box_format="xywh"):
    directory = Path(root).expanduser().resolve() / name
    annotations = [p for p in [directory / "groundtruth.txt", directory / "groundtruth_rect.txt"] if p.is_file()]
    if not annotations:
        raise FileNotFoundError("Missing groundtruth.txt / groundtruth_rect.txt: {}".format(directory))
    raw_boxes = read_boxes(annotations[0])
    if len(annotations) == 2 and not np.array_equal(raw_boxes, read_boxes(annotations[1])):
        raise ValueError("Conflicting annotation files in {}".format(directory))
    boxes = boxes_to_xywh(raw_boxes, box_format)
    candidates = []
    for image_dir in [directory, directory / "img", directory / "imgs"]:
        if image_dir.is_dir():
            frames = sorted([p for p in image_dir.iterdir() if p.is_file() and
                             p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: natural_key(p.name))
            if frames:
                candidates.append(frames)
    if len(candidates) != 1:
        raise ValueError("Expected exactly one image location (sequence/, img/, imgs/) in {}; got {}".format(
            directory, len(candidates)))
    frames = candidates[0]
    if len(frames) != len(boxes):
        raise ValueError("{}: {} images vs {} annotation rows; refusing to truncate or pad".format(
            name, len(frames), len(boxes)))

    if all(p.stem.isdecimal() for p in frames):
        ids = [int(p.stem) for p in frames]
        if len(set(ids)) != len(ids) or any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("Non-consecutive/duplicate numeric frame names in {}".format(directory))
    return SequenceFiles(name, directory, frames, boxes, annotations[0], raw_boxes, str(box_format).strip().lower())


def load_split(root, names=None, decode=False, training=False, box_format="xywh",
               allow_mismatched_train=None, return_skipped=False):


    names = sequence_names(root) if names is None else list(names)
    if allow_mismatched_train is None:
        if training:
            import local_settings as settings
            allow_mismatched_train = bool(getattr(settings, "TRAIN_SKIP_MISMATCHED_SEQUENCES", False))
        else:
            allow_mismatched_train = False
    if not training and allow_mismatched_train:
        raise ValueError("Skipping mismatched sequences is forbidden for evaluation")
    records, skipped = [], []
    for name in names:
        try:
            record = read_sequence(root, name, box_format=box_format)
        except ValueError as exc:

            match = re.fullmatch(
                re.escape(name) + r": (\d+) images vs (\d+) annotation rows; refusing to truncate or pad",
                str(exc),
            )
            if not (training and allow_mismatched_train and match):
                raise
            skipped.append({
                "sequence": name,
                "images": int(match.group(1)),
                "annotation_rows": int(match.group(2)),
                "reason": "frame_annotation_count_mismatch",
            })
            continue
        records.append(record)
    if skipped:
        import local_settings as settings
        limit = int(getattr(settings, "TRAIN_MAX_SKIPPED_MISMATCHED_SEQUENCES", 5))
        if limit < 0 or len(skipped) > limit:
            raise RuntimeError(
                "{} mismatched TRAIN sequences exceed configured maximum {}: {}. "
                "Repair the raw data before formal training.".format(len(skipped), limit, skipped))
        print("[LSOTB DATA WARNING] Excluded {} mismatched TRAINING sequences. "
              "No frame/GT rows were truncated, padded or shifted.".format(len(skipped)), flush=True)
        for row in skipped:
            print("[LSOTB DATA WARNING] SKIP {sequence}: {images} images, "
                  "{annotation_rows} GT rows; please restore the missing frame.".format(**row), flush=True)
        print("[LSOTB DATA WARNING] Using {} of {} listed sequences. "
              "Training protocol changes until the dataset is repaired.".format(
                  len(records), len(names)), flush=True)
    if not records:
        raise RuntimeError("No valid LSOTB sequences remain")
    sampler_rejections = []
    for record in records:
        valid = (record.boxes[:, 2:] > 0).all(axis=1)
        if training and (len(valid) < 20 or int(valid.sum()) <= 10):


            sampler_rejections.append((record.name, len(valid), int(valid.sum())))
        if not training and not valid[0]:
            raise ValueError("{} has invalid initialization box".format(record.name))
        if decode:
            for frame in record.frames:
                read_rgb(frame)
    if training:
        eligible_count = len(records) - len(sampler_rejections)
        if eligible_count == 0:
            raise ValueError(
                "No LSOTB training sequences satisfy the official sampler "
                "(>=20 frames, >10 valid boxes for 3 templates + 2 searches). "
                "Checked {} sequences under {}. Counts (name, frames, valid_boxes): {}".format(
                    len(records), root, sampler_rejections[:10]))
        if sampler_rejections:
            print("[LSOTB] {} sequences loaded; {} satisfy the official sampler; "
                  "{} will trigger official sequence re-sampling (records retained).".format(
                      len(records), eligible_count, len(sampler_rejections)), flush=True)
            for name, frame_count, valid_count in sampler_rejections[:10]:
                print("[LSOTB] re-sample {}: frames={}, valid_boxes={} "
                      "(requires >=20 frames and >10 valid boxes)".format(
                          name, frame_count, valid_count), flush=True)
            if len(sampler_rejections) > 10:
                print("[LSOTB] {} additional sequences have the same sampling status.".format(
                    len(sampler_rejections) - 10), flush=True)
    return (records, skipped) if return_skipped else records


def read_rgb(path):
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Image decode failed: {}".format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def split_fingerprint(records):
    h = hashlib.sha256()
    for record in records:
        h.update(str(record.directory).encode("utf-8"))
        h.update(record.boxes.astype("<f8").tobytes())
        for frame in record.frames:
            stat = frame.stat()
            h.update("{}:{}:{}\n".format(frame.name, stat.st_size, stat.st_mtime_ns).encode("utf-8"))
    return h.hexdigest()
