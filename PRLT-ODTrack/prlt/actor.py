import torch

from lib.train.actors.odtrack import ODTrackActor
from lib.utils.heapmap_utils import generate_heatmap
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from .ptr import attach_modules
from .objectives import FrameFocal, frame_objectives
from .teacher import paired_teacher_forward
from .temporal import search_features, temporal_target_evidence
from .options import PRLT


@torch.no_grad()
def bbox_iou(pred, gt):
    p = box_cxcywh_to_xyxy(pred[:, 0, :].detach()).clamp(0, 1)
    t = box_xywh_to_xyxy(gt.detach()).clamp(0, 1)
    lt, rb = torch.maximum(p[:, :2], t[:, :2]), torch.minimum(p[:, 2:], t[:, 2:])
    wh = (rb-lt).clamp_min(0)
    intersection = wh[:, 0]*wh[:, 1]
    pa = (p[:, 2]-p[:, 0]).clamp_min(0)*(p[:, 3]-p[:, 1]).clamp_min(0)
    ta = (t[:, 2]-t[:, 0]).clamp_min(0)*(t[:, 3]-t[:, 1]).clamp_min(0)
    return intersection/(pa+ta-intersection+1e-6)


class PRLTActor(ODTrackActor):
    def __init__(self, net, objective, loss_weight, settings, cfg, options=PRLT):
        self.ptr, self.target = attach_modules(net, cfg, options)
        self.options, self._step = options, 0
        self.frame_focal = FrameFocal(objective["focal"])
        super().__init__(net, dict(objective, focal=self.frame_focal), loss_weight, settings, cfg)

    def __call__(self, data):
        if self.net.training:
            self._step += 1
        return super().__call__(data)

    def forward_pass(self, data):
        if not self.net.training:
            return super().forward_pass(data)
        core = getattr(self.net, "module", self.net)
        captured = []
        def observe(module, arguments, output):
            captured.append(output[0].detach())
        handle = core.backbone.blocks[self.options.layer_index].register_forward_hook(observe)
        try:
            outputs = super().forward_pass(data)
        finally:
            handle.remove()
        if len(outputs) != len(captured) or len(outputs) != data["search_images"].shape[0]:
            raise RuntimeError("PRLT captured ODTrack search frames out of order")
        for i, feature in enumerate(captured):
            out = outputs[i]
            out.update(self.ptr(feature, data["search_images"][i], len(data["template_images"]), core.token_len))
            base = out["R_map"]
            features = torch.cat([base.detach(), out["E_map"].detach().abs(), out["score_map"].detach()], dim=1)
            out["R_base"] = base
            out["R_delta"] = self.options.TARGET_DELTA_MAX*torch.tanh(self.target(features))
            out.update(temporal_teacher=None, temporal_mask=None,
                       temporal_good_frac=0., teacher_executed=False)
        o = self.options
        teacher_due = (int(data.get("epoch", 1)) >= o.TARGET_START_EPOCH
                       and self._step % max(1, o.TARGET_VIEW_INTERVAL) == 0
                       and "physics_images" in data)
        if teacher_due:
            templates = [data["template_images"][j] for j in range(len(data["template_images"]))]
            for i, out in enumerate(outputs):
                valid = data["physics_valid"][i].reshape(-1).to(data["search_images"].device)
                ids = torch.where(valid>0.5)[0][:max(1, o.TARGET_SUBSET)]
                if not len(ids):
                    continue
                current, reference = paired_teacher_forward(core, [z[ids] for z in templates],
                    data["search_images"][i][ids], data["physics_images"][i][ids])
                spatial = out["score_map"].shape[-2:]
                now, before = search_features(current, spatial), search_features(reference, spatial)
                if now is None or before is None:
                    raise RuntimeError("ODTrack teacher did not return final search features")
                good = ((bbox_iou(current["pred_boxes"], data["search_anno"][i][ids]) >= o.TARGET_MIN_IOU)
                        & (bbox_iou(reference["pred_boxes"], data["physics_anno"][i][ids]) >= o.TARGET_MIN_IOU))
                target, mask = temporal_target_evidence(now, before, data["search_anno"][i][ids],
                    data["physics_anno"][i][ids], (valid[ids]>0.5).float()*good.float(),
                    delta_max=o.TARGET_DELTA_MAX, roi_size=o.TARGET_ROI_SIZE, radius=o.TARGET_LOCAL_RADIUS,
                    outlier_mad=o.TARGET_OUTLIER_MAD, outlier_abs=o.TARGET_OUTLIER_ABS,
                    tau=o.TARGET_STABILITY_TAU, min_common_fraction=o.ROI_MIN_COMMON_FRACTION)

                def scatter(value):
                    full = data["search_images"][i].new_zeros(valid.numel(), 1, *spatial)
                    return full.index_copy(0, ids, value.to(full))

                out.update(temporal_teacher=scatter(target), temporal_mask=scatter(mask), teacher_executed=True,
                           temporal_good_frac=float((mask.detach().flatten(1).sum(1)>0).float().sum().item())/valid.numel())
        if core.track_query is not None:
            raise RuntimeError("PRLT teacher leaked a query into the next independent minibatch")
        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if not self.net.training:
            return super().compute_losses(pred_dict, gt_dict, return_status)
        maps = generate_heatmap(gt_dict["search_anno"], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        weights, auxiliary, frame_stats = [], [], []
        for i, out in enumerate(pred_dict):
            weight, loss, status = frame_objectives(out, gt_dict, i, maps[i].unsqueeze(1), self.cfg, self.options)
            weights.append(weight)
            auxiliary.append(loss)
            frame_stats.append(status)

        with self.frame_focal.bind(weights):
            base, status = super().compute_losses(pred_dict, gt_dict, True)
        aux = sum(auxiliary)
        total = base+aux
        for i, frame in enumerate(frame_stats):
            status.update({"PRLT/{}frame_{}".format(i, key): value for key, value in frame.items()})
        status.update({"PRLT/step": self._step, "PRLT/alpha": frame_stats[0]["alpha"],
                       "PRLT/base_tracking": float(base.detach()), "PRLT/aux_total": float(aux.detach()),
                       "Loss/total": float(total.detach())})
        return (total, status) if return_status else total
