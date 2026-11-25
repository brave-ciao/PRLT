from contextlib import contextmanager
import random
import numpy as np
import torch
from torch.nn import functional as F

from lib.utils import TensorDict
from lib.train.data.sampler import TrackingSampler
from lib.train.data.processing import STARKProcessing
from lib.train.data import processing_utils as prutils, LTRLoader
from lib.train.base_functions import build_dataloaders as original_dataloaders
from .options import PRLT


def cpu_rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def set_cpu_rng_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


@contextmanager
def replay_rng(state):
    caller = cpu_rng_state()
    try:
        set_cpu_rng_state(state)
        yield
    finally:
        set_cpu_rng_state(caller)


class RecordingTransform:
    def __init__(self, transform):
        self.transform, self.state = transform, None

    def __call__(self, **inputs):
        self.state = cpu_rng_state()
        return self.transform(**inputs)


class PRLTProcessing(STARKProcessing):
    @staticmethod
    def transfer_jitter(source_box, source_jittered, target_box):
        src_wh = source_box[2:4].clamp_min(0.0001)
        src_c = source_box[0:2]+0.5*src_wh
        jit_wh = source_jittered[2:4].clamp_min(0.0001)
        jit_c = source_jittered[0:2]+0.5*jit_wh
        scale, offset = jit_wh/src_wh, (jit_c-src_c)/src_wh
        tgt_wh = target_box[2:4].clamp_min(0.0001)
        tgt_c = target_box[0:2]+0.5*tgt_wh
        out_wh, out_c = tgt_wh*scale, tgt_c+offset*tgt_wh
        return torch.cat((out_c-0.5*out_wh, out_wh), dim=0)

    def _get_jittered_box(self, box, mode):
        jittered = super()._get_jittered_box(box, mode)
        if mode == "search" and hasattr(self, "_reference_jitter"):
            self._reference_jitter.append((box.clone(), jittered.clone()))
        return jittered

    def __call__(self, data):
        if "physics_images" not in data:
            return super().__call__(data)
        refs = {k: data.pop(k) for k in ("physics_images", "physics_anno", "physics_delta", "physics_valid")}
        if len(refs["physics_images"]) != len(data["search_images"]):
            raise ValueError("Every ODTrack search must have exactly one PRLT reference")
        self._reference_jitter = []
        original = self.transform["search"]
        recorder = RecordingTransform(original)
        self.transform["search"] = recorder
        try:
            data = super().__call__(data)
        finally:
            self.transform["search"] = original
        if not data["valid"]:
            return data
        images, boxes = refs["physics_images"], refs["physics_anno"]
        masks = [torch.zeros(im.shape[:2]) for im in images]
        if self.transform["joint"] is not None:
            images, boxes, masks = self.transform["joint"](image=images, bbox=boxes, mask=masks, new_roll=False)
        jittered = [self.transfer_jitter(src, jit, ref) for (src, jit), ref in zip(self._reference_jitter, boxes)]
        crops, crop_boxes, att, crop_masks = prutils.jittered_center_crop(images, jittered, boxes,
            self.search_area_factor["search"], self.output_sz["search"], masks=masks)
        with replay_rng(recorder.state):
            images, boxes, att, _ = original(image=crops, bbox=crop_boxes, att=att, mask=crop_masks, joint=False)
        valid = torch.stack([torch.as_tensor(v).reshape(()) for v in refs["physics_valid"]]).float()
        for i, attention in enumerate(att):
            down = F.interpolate(attention[None, None].float(), size=self.output_sz["search"]//16).bool()
            if bool(attention.all()) or bool(down.all()):
                valid[i] = 0.
        data.update(physics_images=torch.stack(images), physics_anno=torch.stack(boxes),
                    physics_delta=torch.stack([torch.as_tensor(v).reshape(()) for v in refs["physics_delta"]]).float(),
                    physics_valid=valid)
        self._reference_jitter = []
        return data


class PRLTTrackingSampler(TrackingSampler):
    def __init__(self, *args, options=PRLT, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = options
        if self.frame_sample_mode != "causal" or self.train_cls:
            raise ValueError("This PRLT data adapter preserves the verified ODTrack causal tracking sampler")

    def reference_id(self, visible, search_id):
        candidates = [i for i in range(max(0, search_id-self.options.PHYSICS_REF_MAX_GAP), search_id) if bool(visible[i])]
        return (candidates[-1], search_id-candidates[-1], 1.) if candidates else (search_id, 1, 0.)

    def getitem(self):
        valid = False
        while not valid:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video = dataset.is_video_sequence()
            seq_id, visible, info = self.sample_seq_from_dataset(dataset, is_video)
            if is_video:
                search_ids, gap = None, 0
                while search_ids is None:
                    base = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames-1,
                                                    max_id=len(visible)-self.num_search_frames)
                    previous = self._sample_visible_ids(visible, num_ids=self.num_template_frames-1,
                        min_id=base[0]-self.max_gap-gap, max_id=base[0])
                    if previous is None:
                        gap += 5
                        continue
                    template_ids = base+previous
                    if len(template_ids)>1:
                        template_ids.sort()
                    search_ids = self._sample_visible_ids(visible, num_ids=self.num_search_frames,
                        min_id=template_ids[-1]+1, max_id=template_ids[-1]+self.max_gap+gap)
                    if search_ids is not None and len(search_ids)>1:
                        search_ids.sort()
                    gap += 5
            else:
                template_ids = [1]*self.num_template_frames
                search_ids = [1]*self.num_search_frames
            try:
                templates, ta, tm = dataset.get_frames(seq_id, template_ids, info)
                searches, sa, sm = dataset.get_frames(seq_id, search_ids, info)
                h, w, _ = templates[0].shape
                data = TensorDict({"template_images": templates, "template_anno": ta["bbox"],
                    "template_masks": ta.get("mask", [torch.zeros((h, w))]*self.num_template_frames),
                    "search_images": searches, "search_anno": sa["bbox"],
                    "search_masks": sa.get("mask", [torch.zeros((h, w))]*self.num_search_frames),
                    "dataset": dataset.get_name(), "test_class": sm.get("object_class_name"),
                    "exp_str": tm.get("exp_str")})

                reference = [self.reference_id(visible, i) if is_video else (i, 1, 0.) for i in search_ids]
                frames, anno, _ = dataset.get_frames(seq_id, [item[0] for item in reference], info)
                data.update(physics_images=frames, physics_anno=anno["bbox"],
                    physics_delta=[torch.tensor(float(item[1])) for item in reference],
                    physics_valid=[torch.tensor(item[2]) for item in reference])
                data = self.processing(data)
                valid = data["valid"]
            except (FileNotFoundError, OSError):
                raise
        return data


def build_dataloaders(cfg, settings):
    train, validation = original_dataloaders(cfg, settings)
    source, p = train.dataset, train.dataset.processing
    processing = PRLTProcessing(search_area_factor=p.search_area_factor, output_sz=p.output_sz,
        center_jitter_factor=p.center_jitter_factor, scale_jitter_factor=p.scale_jitter_factor,
        mode=p.mode, settings=p.settings, transform=p.transform["template"],
        search_transform=p.transform["search"], joint_transform=p.transform["joint"])
    extended = PRLTTrackingSampler(datasets=source.datasets, p_datasets=source.p_datasets,
        samples_per_epoch=source.samples_per_epoch, max_gap=source.max_gap,
        num_search_frames=source.num_search_frames, num_template_frames=source.num_template_frames,
        processing=processing, frame_sample_mode=source.frame_sample_mode, train_cls=source.train_cls,
        pos_prob=source.pos_prob)
    if settings.local_rank != -1:
        raise ValueError("This PRLT delivery uses one GPU")
    train = LTRLoader("train", extended, training=True, batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True, num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1)
    return train, validation
