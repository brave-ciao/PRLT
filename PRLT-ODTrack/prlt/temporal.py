import torch
import torch.nn.functional as F
from .physics import crop_roi, paste_roi, aligned_roi_boxes


def search_features(out, hw):

    tokens = out.get("backbone_feat")
    if isinstance(tokens, (list, tuple)):
        tokens = tokens[-1]
    if tokens is None or tokens.ndim != 3:
        return None
    h, w = map(int, hw)
    if tokens.shape[1] < h * w:
        return None
    return tokens[:, -h * w :, :].transpose(1, 2).reshape(tokens.size(0), tokens.size(2), h, w)


@torch.no_grad()
def temporal_target_evidence(
    now_feat,
    ref_feat,
    now_boxes,
    ref_boxes,
    valid,
    delta_max=0.1,
    roi_size=8,
    radius=1,
    outlier_mad=2.0,
    outlier_abs=0.03,
    tau=0.08,
    min_common_fraction=0.5,
):


    if now_feat.shape != ref_feat.shape:
        raise ValueError("Temporal teacher feature shapes must match")
    if roi_size < 3 or radius < 0:
        raise ValueError("roi_size>=3, radius>=0 required")
    b, c, h, w = now_feat.shape
    dtype = now_feat.dtype
    br, bx, roi_valid, _ = aligned_roi_boxes(
        ref_boxes.float(), now_boxes.float(), 0.96, min_common_fraction=min_common_fraction
    )
    current = F.normalize(
        crop_roi(now_feat.detach().float(), bx.float(), roi_size), dim=1, eps=1e-06
    )
    former = F.normalize(
        crop_roi(ref_feat.detach().float(), br.float(), roi_size), dim=1, eps=1e-06
    )
    k = radius * 2 + 1
    patches = F.unfold(F.pad(former, (radius,) * 4, mode="replicate"), kernel_size=k)
    patches = patches.reshape(b, c, k * k, roi_size * roi_size)
    current = current.reshape(b, c, 1, roi_size * roi_size)
    best = (current * patches).sum(1).amax(1).reshape(b, 1, roi_size, roi_size).clamp(-1.0, 1.0)
    mismatch = (1.0 - best).clamp(0, 2)
    median = mismatch.flatten(1).median(dim=1).values.view(b, 1, 1, 1)
    mad = (
        (mismatch - median).abs().flatten(1).median(dim=1).values.view(b, 1, 1, 1).clamp_min(0.001)
    )
    threshold = median + float(outlier_mad) * mad + float(outlier_abs)
    excess = (mismatch - threshold).clamp_min(0)
    risk = (1.0 - torch.exp(-excess / max(float(tau), 1e-05))).clamp(0, 1)
    is_valid = valid.detach().reshape(-1, 1, 1, 1).to(risk).clamp(0, 1)
    is_valid = is_valid * roi_valid.to(risk).view(b, 1, 1, 1)
    teacher_roi = -float(delta_max) * risk * is_valid
    teacher_map, mask = paste_roi(teacher_roi, bx.float(), h, w)
    return teacher_map.to(dtype), mask.to(dtype) * is_valid.to(dtype)


def calibrated_r_inside_box(base, predicted_delta, boxes, detach_delta_for_prr=True):

    if predicted_delta is None:
        return (base, torch.zeros_like(base), torch.zeros_like(base))
    b, _, h, w = base.shape
    ys = (torch.arange(h, device=base.device, dtype=base.dtype) + 0.5) / h
    xs = (torch.arange(w, device=base.device, dtype=base.dtype) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = (
        (
            (xx[None] >= boxes[:, 0, None, None])
            & (xx[None] <= boxes[:, 0, None, None] + boxes[:, 2, None, None])
            & (yy[None] >= boxes[:, 1, None, None])
            & (yy[None] <= boxes[:, 1, None, None] + boxes[:, 3, None, None])
        )
        .unsqueeze(1)
        .to(base.dtype)
    )
    applied_delta = predicted_delta * mask
    corrected = (base + (applied_delta.detach() if detach_delta_for_prr else applied_delta)).clamp(
        0.0, 1.0
    )
    return (corrected, applied_delta, mask)


def temporal_delta_loss(delta_pred, teacher_delta, mask, easy_weight=0.25, change_threshold=0.007):

    if delta_pred.shape != teacher_delta.shape or delta_pred.shape != mask.shape:
        raise ValueError("Temporal delta loss map dimensions mismatch")
    changed = (teacher_delta.abs() > change_threshold).to(mask.dtype) * mask
    stable = (mask - changed).clamp_min(0)
    error = F.smooth_l1_loss(delta_pred, teacher_delta, beta=0.025, reduction="none")
    l_change = (error * changed).sum() / changed.sum().clamp_min(1.0)
    l_stable = (error * stable).sum() / stable.sum().clamp_min(1.0)
    return l_change + float(easy_weight) * l_stable
