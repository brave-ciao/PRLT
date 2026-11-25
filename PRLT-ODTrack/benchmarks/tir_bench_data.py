from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np

SPECS = Path(__file__).resolve().parent / "tir_benchmark_specs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def rectangles(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = re.split(r"[,;\s]+", line.strip())
        if len(fields) != 4:
            raise ValueError("{}:{} needs four xywh columns, found {}".format(path, number, len(fields)))
        rows.append([float(v) for v in fields])
    result = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Empty/non-finite rectangle file: " + str(path))
    return result


def numeric_digest(boxes):
    return hashlib.sha256(np.asarray(boxes, dtype="<f8").tobytes()).hexdigest()


def numeric_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Missing IR image directory: " + str(directory))
    frames = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not frames:
        raise FileNotFoundError("No image files in " + str(directory))
    if not all(p.stem.isdecimal() for p in frames):
        raise ValueError("Expected numeric frame names in " + str(directory))
    frames.sort(key=lambda p: int(p.stem))
    ids = np.asarray([int(p.stem) for p in frames], dtype=np.int64)
    if len(ids) > 1 and not np.all(np.diff(ids) == 1):
        raise ValueError("Missing/duplicate/non-consecutive frame numbers in {}. "
                         "Full-rate IR images are required; do not keep only annotated frames.".format(directory))
    return frames, ids


@dataclass
class Record:
    name: str
    split: str
    directory: Path
    frames: list
    boxes: np.ndarray
    annotation_path: Path
    score_indices: np.ndarray
    frame_ids: np.ndarray

    def __post_init__(self):
        if len(self.frames) < 2 or len(self.boxes) != len(self.score_indices):
            raise ValueError("Invalid frame/annotation mapping for " + self.name)
        if self.score_indices[0] != 0 or self.score_indices[-1] >= len(self.frames):
            raise ValueError("Annotations do not map to available frames: " + self.name)
        if not np.all(self.boxes[0, 2:] > 0):
            raise ValueError("First annotation has no valid initialization box: " + self.name)


def load_protocol(benchmark):
    if benchmark not in ("ptbtir", "vtuavtir"):
        raise ValueError("Unknown benchmark: " + str(benchmark))
    protocol = json.loads((SPECS / (benchmark + "_protocol.json")).read_text(encoding="utf-8"))
    records = protocol["records"]
    expected = 60 if benchmark == "ptbtir" else 250
    if len(records) != expected or len({r["name"] for r in records}) != expected:
        raise ValueError("Corrupt official benchmark name list")
    for r in records:
        n = r["name"]
        if not n or n in (".", "..") or Path(n).name != n or "/" in n or "\\" in n:
            raise ValueError("Invalid sequence name: " + repr(n))
    if benchmark == "ptbtir":
        if sum(r["frame_count"] for r in records) != 30029:
            raise ValueError("Official PTB-TIR contains 30,029 frames")
    elif (sum(r["split"] == "st" for r in records), sum(r["split"] == "lt" for r in records)) != (176, 74):
        raise ValueError("Official VTUAV test split must contain ST176/LT74")
    return protocol


def ptb_directories(root, specs):

    names = {spec["name"] for spec in specs}
    direct = {}
    for name in sorted(names):
        candidates = (root / name, root / "tirsequences" / name)
        found = list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
        if len(found) > 1:
            raise ValueError(f"PTB-TIR {name} has multiple folders: {found}. Set PTBTIR_ROOT to one dataset.")
        if found:
            direct[name] = found[0]
    if len(direct) == len(names):
        return direct


    canonical = {name.casefold(): name for name in names}
    max_wrapper_depth = 4
    parents, visited = [root], set()
    best_parent, best_names = root, set(direct)
    for _ in range(max_wrapper_depth + 1):
        matches, children = [], []
        for parent in parents:
            parent = parent.resolve()
            if parent in visited:
                continue
            visited.add(parent)
            locations = {}
            for entry in parent.iterdir():
                if not entry.is_dir():
                    continue
                name = canonical.get(entry.name.casefold())
                if name is None:
                    children.append(entry)
                else:
                    locations.setdefault(name, set()).add(entry.resolve())
            if len(locations) > len(best_names):
                best_parent, best_names = parent, set(locations)
            if len(locations) == len(names):
                duplicates = sorted(name for name, paths in locations.items() if len(paths) > 1)
                if duplicates:
                    raise ValueError(f"PTB-TIR has duplicate sequence folders in {parent}: {duplicates}.")
                matches.append((parent, {name: next(iter(paths)) for name, paths in locations.items()}))
        if len(matches) > 1:
            paths = ", ".join(str(parent) for parent, _ in sorted(matches, key=lambda item: item[0]))
            raise ValueError(f"PTB-TIR has multiple complete sequence roots: {paths}. Set PTBTIR_ROOT explicitly.")
        if matches:
            parent, directories = matches[0]
            print(f"PTB-TIR sequence parent: {parent}", flush=True)
            return directories
        parents = children

    missing = ", ".join(sorted(names - best_names)[:5])
    raise FileNotFoundError(
        f"PTBTIR_ROOT={root} does not contain the complete PTB-TIR image dataset. "
        f"Best match: {best_parent} ({len(best_names)}/{len(names)} sequences, missing {missing}). "
        "Set PTBTIR_ROOT to the directory containing the sequence folders.")


def ptb_record(directory, spec, annotation_root=""):
    name = spec["name"]
    image_dirs = [p for p in (directory / "img", directory / "imgs", directory)
                  if p.is_dir() and any(f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir())]
    if len(image_dirs) != 1:
        raise ValueError("{}: expected one image location (img/, imgs/, sequence/), found {}".format(name, len(image_dirs)))
    frames, ids = numeric_images(image_dirs[0])
    if not np.array_equal(ids, np.arange(spec["start"], spec["end"] + 1)):
        raise ValueError("{}: frame IDs/count differ from official PTB-TIR {}..{} ({} frames)".format(
            name, spec["start"], spec["end"], spec["frame_count"]))
    if annotation_root:
        annotation = Path(annotation_root).expanduser().resolve() / (name + ".txt")
    else:
        paths = [directory / filename for filename in ("groundtruth.txt", "groundtruth_rect.txt", name + ".txt")]
        annotation = next((path for path in paths if path.is_file()), None)
        if annotation is None:
            raise FileNotFoundError(f"{name}: set PTBTIR_ANNOTATION_ROOT to the official toolkit's annotation directory.")
    boxes = rectangles(annotation)
    if len(boxes) != spec["frame_count"] or numeric_digest(boxes) != spec["annotation_xywh_sha256"]:
        raise ValueError(f"{name}: use the corrected official PTB-TIR annotations via PTBTIR_ANNOTATION_ROOT.")
    return Record(name, "ope", directory, frames, boxes, annotation, np.arange(len(frames)), ids)


def vtuav_record(directory, name, split, image_subdir="ir"):
    if not image_subdir or Path(image_subdir).name != image_subdir or image_subdir in (".", "..", "rgb"):
        raise ValueError("IMAGE_SUBDIR must name one IR folder (normally 'ir')")
    directory = Path(directory).expanduser().resolve()

    frames, ids = numeric_images(directory / image_subdir)
    annotation = directory / "ir.txt"
    boxes = rectangles(annotation)
    indices = np.arange(0, len(frames), 10, dtype=np.int64)
    if len(indices) != len(boxes):
        raise ValueError("{}: {} full-rate IR images -> {} annotation positions (1,11,21,...), "
                         "but ir.txt has {} rows. Refusing to truncate, pad, interpolate GT, "
                         "or change the tracking stride.".format(name, len(frames), len(indices), len(boxes)))
    return Record(name, split, directory, frames, boxes, annotation, indices, ids)


class BenchmarkData:
    def __init__(self, benchmark, options):
        self.benchmark = benchmark
        self.options = options
        self.protocol = load_protocol(benchmark)
        self.root = Path(options.DATA_ROOT).expanduser().resolve()
        specs = self.protocol["records"]
        if benchmark == "ptbtir":
            if not str(options.DATA_ROOT).strip() or not self.root.is_dir():
                raise FileNotFoundError("Set PTBTIR_ROOT in local_settings.py to an existing "
                                        f"image dataset directory. Current path: {self.root}")
            self.split = "ope"
            directories = ptb_directories(self.root, specs)
            self.records = [ptb_record(directories[s["name"]], s, options.ANNOTATION_ROOT) for s in specs]
        else:
            self.split = str(options.SPLIT).lower()
            if self.split not in ("st", "lt", "all_test"):
                raise ValueError("SPLIT must be 'st', 'lt', or 'all_test'; training sequences are never evaluated")
            specs = [s for s in specs if self.split == "all_test" or s["split"] == self.split]
            needed_splits = {s["split"] for s in specs}
            if (not str(options.DATA_ROOT).strip() or not self.root.is_dir()) and not all(
                    getattr(options, s.upper() + "_ROOT", "") for s in needed_splits):
                raise FileNotFoundError("Set VTUAVTIR_ROOT or VTUAVTIR_ST_ROOT in local_settings.py")
            paths, missing = {}, []
            for spec in specs:
                name, split = spec["name"], spec["split"]
                override = getattr(options, split.upper() + "_ROOT", "")
                candidates = ([Path(override).expanduser() / name] if override else
                              [self.root / name, self.root / ("test_" + split.upper()) / name,
                               self.root / split.upper() / name,
                               *self.root.glob(f"test_{split.upper()}_*/{name}"),
                               *(self.root / "short-term").glob(f"test_{split.upper()}_*/{name}")])
                present = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
                if len(present) > 1:
                    raise ValueError("Duplicate VTUAV sequence locations: {}. Set {}_ROOT explicitly.".format(present, split.upper()))
                if not present:
                    missing.append(name)
                else:
                    paths[name] = present[0]
            if missing:
                raise FileNotFoundError("VTUAV-TIR {} is missing {} official sequences:\n{}".format(
                    self.split, len(missing), "\n".join(missing)))
            self.records = [vtuav_record(paths[s["name"]], s["name"], s["split"], options.IMAGE_SUBDIR) for s in specs]
        self.names = [r.name for r in self.records]
        self.identity = self._fingerprint()

    def _fingerprint(self):
        h = hashlib.sha256()
        h.update(self.benchmark.encode())
        h.update(self.split.encode())
        for record in self.records:
            h.update(str(record.directory).encode("utf-8"))
            h.update(record.name.encode("utf-8"))
            h.update(record.split.encode())
            h.update(record.boxes.astype("<f8").tobytes())
            h.update(record.score_indices.astype("<i8").tobytes())
            for p in record.frames:
                s = p.stat()
                h.update("{}:{}:{}\n".format(p, s.st_size, s.st_mtime_ns).encode("utf-8"))
        return h.hexdigest()


def read_ir(path):


    import cv2
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("Image decode failed: " + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
