from collections import OrderedDict
import torch

from .base_video_dataset import BaseVideoDataset
from prlt.lsotb import load_split, read_rgb


class LSOTB(BaseVideoDataset):
    """LSOTB data interface for the unchanged ODTrack causal sampler."""
    def __init__(self, root, image_loader=None, box_format="xywh"):
        super().__init__("LSOTB", str(root), image_loader or read_rgb)
        self.records = load_split(root, training=True, box_format=box_format)
        self.box_format = box_format
        self.sequence_list = [r.name for r in self.records]

    def get_name(self):
        return "lsotb"

    def get_sequence_info(self, seq_id):
        bbox = torch.as_tensor(self.records[seq_id].boxes.copy(), dtype=torch.float32)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {"bbox": bbox, "valid": valid, "visible": valid.byte()}

    def get_frames(self, seq_id, frame_ids, anno=None):
        record = self.records[seq_id]
        frames = [read_rgb(record.frames[int(frame_id)]) for frame_id in frame_ids]
        anno = self.get_sequence_info(seq_id) if anno is None else anno
        annotations = {key: [value[int(i)].clone() for i in frame_ids] for key, value in anno.items()}
        metadata = OrderedDict(object_class_name=None, motion_class=None, major_class=None,
                               root_class=None, motion_adverb=None)
        return frames, annotations, metadata
