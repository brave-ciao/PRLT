from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

BENCHMARK = "lashertir"
ANNOTATION_POLICY = "aligned_lasher_complete_init_xywh_v2"
REFERENCE = Path(__file__).resolve().parent / "lashertir_reference"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def numeric_digest(array):
    values = np.asarray(array, dtype="<f8").copy(order="C")
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(str(values.shape).encode("ascii") + values.tobytes()).hexdigest()


def official_test_names():
    path = REFERENCE / "testingsetList.txt"
    names = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    metadata = json.loads((REFERENCE / "protocol.json").read_text(encoding="utf-8"))
    if len(names) != 245 or len(set(names)) != 245 or digest(names) != metadata["test_names_digest"]:
        raise ValueError("Bundled official LasHeR 245-name test manifest is inconsistent")
    if any(name in (".", "..") or "/" in name or "\\" in name for name in names):
        raise ValueError("Invalid sequence name in LasHeR manifest")
    return names


def rectangles(path, frame_count=None):

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
        value = np.loadtxt(io.StringIO(text.replace(",", " ").replace(";", " ")),
                           dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as e:
        raise ValueError("Cannot read xywh rectangles {}: {}".format(path, e)) from e
    if value.ndim != 2 or value.shape[1] != 4 or not len(value) or np.isinf(value).any():
        raise ValueError("Expected nonempty N x 4 xywh, without Inf: {}; got {}".format(path, value.shape))
    if frame_count is not None and len(value) != frame_count:
        raise ValueError("{} has {} rows; expected {} (one row per IR frame, including initialization)".format(
            path, len(value), frame_count))
    return value


def resolve_root(path, names):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LasHeR root does not exist: {root}. Set LASHERTIR_ROOT in local_settings.py.")
    candidates = list(dict.fromkeys([root, root / "testingset", root / "TestingSet" / "testingset",
                                    root / "TestingSet", root / "LasHeR_testingset"]))
    counts = [(p, sum((p / name).is_dir() for name in names)) for p in candidates if p.is_dir()]
    complete = [p for p, count in counts if count == len(names)]
    if len(complete) > 1:
        raise ValueError("Multiple complete LasHeR roots; set DATA_ROOT to one sequence parent: " + str(complete))
    if complete:
        return complete[0]
    best, count = max(counts, key=lambda pair: pair[1])
    missing = [name for name in names if not (best / name).is_dir()]
    raise FileNotFoundError("Incomplete LasHeR test split at {}: {}/{} sequence folders. Missing: {}".format(
        best, count, len(names), ", ".join(missing[:20])))


def ordered_ir_frames(directory):
    if not Path(directory).is_dir():
        raise FileNotFoundError("Missing IR image folder: {}. Set IMAGE_SUBDIR if renamed.".format(directory))
    items = []
    for p in Path(directory).iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.search(r"(\d+)$", p.stem)
        if match is None:
            raise ValueError("IR image filename needs a numeric suffix for unambiguous frame order: " + str(p))
        items.append((int(match.group(1)), p))
    items.sort(key=lambda item: item[0])
    ids = [i for i, _ in items]
    if not items or len(ids) != len(set(ids)):
        raise ValueError("Empty IR folder or duplicate numeric frame IDs: " + str(directory))
    return tuple(str(p) for _, p in items), tuple(ids)


@dataclass
class Record:
    name: str
    boxes: np.ndarray
    annotation_path: Path
    frames: tuple
    frame_ids: tuple


class LasHeRTIRData:
    def __init__(self, options, require_images=True, names=None):
        self.options = options
        official = official_test_names()

        self.names = official if names is None else list(names)
        if not self.names or len(self.names) != len(set(self.names)) or not set(self.names) <= set(official):
            raise ValueError("Requested LasHeR names must be unique members of the official test split")
        self.root = resolve_root(options.DATA_ROOT, self.names)
        self.records = []
        annotations = Path(options.ANNOTATION_ROOT).expanduser().resolve() if options.ANNOTATION_ROOT else None
        filename = str(getattr(options, "ANNOTATION_FILENAME", "init.txt") or "init.txt")
        self.annotation_policy = (ANNOTATION_POLICY if annotations is None and filename == "init.txt"
                                  else "explicit_annotation_override_xywh_v2")
        for name in self.names:
            anno = annotations / (name + ".txt") if annotations else self.root / name / filename
            if not anno.is_file():
                raise FileNotFoundError("Missing aligned LasHeR annotation: {}. The benchmark's complete GT is "
                                        "init.txt (xywh); infrared.txt/visible.txt are not automatic substitutes. "
                                        "Use ANNOTATION_ROOT only for a verified separate annotation copy.".format(anno))
            boxes = rectangles(anno)
            if not np.isfinite(boxes[0]).all() or np.any(boxes[0, 2:] <= 0):
                raise ValueError("Initial LasHeR GT must be a finite xywh box with positive size: " + str(anno))
            frames, ids = ordered_ir_frames(self.root / name / options.IMAGE_SUBDIR) if require_images else ((), ())
            if require_images and len(frames) != len(boxes):
                raise ValueError("{}: {} IR images vs {} rows in {}; no truncation, padding or frame skipping "
                                 "is allowed".format(name, len(frames), len(boxes), anno))
            self.records.append(Record(name, boxes, anno, frames, ids))
        self.complete_official_split = self.names == official
        self.annotation_identity = digest([{ "name": r.name, "frames": len(r.boxes),
                                              "xywh_digest": numeric_digest(r.boxes)} for r in self.records])
        self.image_identity = None
        if require_images:
            self.image_identity = digest([{"name": r.name,
                "files": [[Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in r.frames]}
                for r in self.records])
