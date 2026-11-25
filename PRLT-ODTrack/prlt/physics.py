import torch
import torch.nn.functional as F


def denorm_gray(image, mean, std):
    c = image.shape[1]
    m = image.new_tensor(list(mean)[:c]).view(1, c, 1, 1)
    s = image.new_tensor(list(std)[:c]).view(1, c, 1, 1)
    x = image * s + m
    return x.mean(1, keepdim=True) if c != 1 else x


def shrink_box(box, ratio):


    if box.ndim != 2 or box.shape[1] != 4:
        raise ValueError("Expected boxes of shape (B, 4) in xywh format")
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("ROI shrink ratio must be in (0, 1]")
    center = box[:, :2] + box[:, 2:] * 0.5
    wh = box[:, 2:] * float(ratio)
    xy = center - 0.5 * wh
    return torch.cat([xy, wh], 1)


@torch.no_grad()
def aligned_roi_boxes(box_ref, box_search, ratio=0.9, min_common_fraction=0.5):


    if box_ref.shape != box_search.shape:
        raise ValueError("Reference/search boxes must have matching shapes")
    if box_ref.ndim != 2 or box_ref.shape[1] != 4:
        raise ValueError("Expected paired boxes with shape (B, 4)")
    if not 0.0 <= float(min_common_fraction) <= 1.0:
        raise ValueError("ROI_MIN_COMMON_FRACTION must be in [0, 1]")
    eps = 1e-06
    finite = torch.isfinite(box_ref).all(1) & torch.isfinite(box_search).all(1)
    positive = (box_ref[:, 2:] > eps).all(1) & (box_search[:, 2:] > eps).all(1)
    input_valid = finite & positive
    fallback = box_ref.new_tensor([0.25, 0.25, 0.5, 0.5]).expand_as(box_ref)
    ref = torch.where(input_valid[:, None], box_ref, fallback)
    cur = torch.where(input_valid[:, None], box_search.to(box_ref), fallback)
    br, bc = (shrink_box(ref, ratio), shrink_box(cur, ratio))
    lo_r = (-br[:, :2] / br[:, 2:].clamp_min(eps)).clamp(0.0, 1.0)
    lo_c = (-bc[:, :2] / bc[:, 2:].clamp_min(eps)).clamp(0.0, 1.0)
    hi_r = ((1.0 - br[:, :2]) / br[:, 2:].clamp_min(eps)).clamp(0.0, 1.0)
    hi_c = ((1.0 - bc[:, :2]) / bc[:, 2:].clamp_min(eps)).clamp(0.0, 1.0)
    lo = torch.maximum(lo_r, lo_c)
    hi = torch.minimum(hi_r, hi_c)
    span = (hi - lo).clamp_min(0.0)
    common = span.prod(1)
    valid = input_valid & (span > eps).all(1) & (common + eps >= float(min_common_fraction))
    rr = torch.cat([br[:, :2] + lo * br[:, 2:], br[:, 2:] * span], 1)
    rc = torch.cat([bc[:, :2] + lo * bc[:, 2:], bc[:, 2:] * span], 1)
    rr = torch.where(valid[:, None], rr, fallback)
    rc = torch.where(valid[:, None], rc, fallback)
    return (rr, rc, valid, torch.where(input_valid, common, torch.zeros_like(common)))


def roi_grid(box, size, device, dtype):
    b = box.shape[0]
    t = (torch.arange(size, device=device, dtype=dtype) + 0.5) / float(size)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    xx, yy = (xx[None], yy[None])
    gx = box[:, 0:1, None] + xx * box[:, 2:3, None]
    gy = box[:, 1:2, None] + yy * box[:, 3:4, None]
    return torch.stack([2 * gx - 1, 2 * gy - 1], -1)


def crop_roi(image, box, size):
    return F.grid_sample(
        image,
        roi_grid(box, size, image.device, image.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def paste_roi(roi, box, h, w):
    b = roi.shape[0]
    xs = (torch.arange(w, device=roi.device, dtype=roi.dtype) + 0.5) / w
    ys = (torch.arange(h, device=roi.device, dtype=roi.dtype) + 0.5) / h
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx[None].expand(b, -1, -1)
    yy = yy[None].expand(b, -1, -1)
    u = (xx - box[:, 0, None, None]) / box[:, 2, None, None].clamp_min(0.0001)
    v = (yy - box[:, 1, None, None]) / box[:, 3, None, None].clamp_min(0.0001)
    grid = torch.stack([2 * u - 1, 2 * v - 1], -1)
    mask = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).unsqueeze(1).to(roi.dtype)
    warped = F.grid_sample(roi, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (warped * mask, mask)


def norm_pair(a, b):
    pair = torch.cat([a, b], dim=-1)
    mean = pair.mean(dim=(-1, -2), keepdim=True)
    std = pair.std(dim=(-1, -2), keepdim=True).clamp_min(0.0001)
    return ((a - mean) / std, (b - mean) / std)


def laplacian(x):
    k = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)


def masked_norm(x):
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + 1e-05)


def compute_physics_maps(
    reference, search, box_ref, box_search, delta, valid, cfg, mean, std,
):


    with torch.no_grad():
        b = search.size(0)
        ref = denorm_gray(reference, mean, std)
        cur = denorm_gray(search, mean, std)
        size = int(getattr(cfg, "ROI_SIZE", 16))
        ratio = float(getattr(cfg, "BOX_SHRINK", 0.9))
        br, bc, roi_valid, common_fraction = aligned_roi_boxes(
            box_ref,
            box_search,
            ratio,
            min_common_fraction=float(getattr(cfg, "ROI_MIN_COMMON_FRACTION", 0.5)),
        )
        tr = crop_roi(ref, br, size)
        tc = crop_roi(cur, bc, size)
        tr, tc = norm_pair(tr, tc)
        med_r = tr.flatten(2).median(dim=-1).values[:, :, None, None]
        med_c = tc.flatten(2).median(dim=-1).values[:, :, None, None]
        mad_r = (
            (tr - med_r).abs().flatten(2).median(dim=-1).values[:, :, None, None].clamp_min(0.0001)
        )
        mad_c = (
            (tc - med_c).abs().flatten(2).median(dim=-1).values[:, :, None, None].clamp_min(0.0001)
        )
        gain = (mad_c / mad_r).clamp(0.8, 1.25)
        tr = (tr - med_r) * gain + med_c
        t_forecast = tr
        steps = (
            delta.reshape(-1)
            .to(search.device)
            .clamp(min=1, max=int(getattr(cfg, "PHYSICS_REF_MAX_GAP", 3)))
        )
        k = float(getattr(cfg, "KAPPA_FIXED", 0.08))
        if not 0 < k < 0.25:
            raise ValueError("KAPPA_FIXED must be in (0, 0.25)")
        for step in range(int(getattr(cfg, "PHYSICS_REF_MAX_GAP", 3))):
            active = (steps > step).to(tc.dtype).view(b, 1, 1, 1)
            t_forecast = t_forecast + active * k * laplacian(t_forecast)
        resid = (tc - t_forecast).abs()
        stable = torch.exp(-resid).clamp(0, 1)
        threshold = float(getattr(cfg, "PHYS_STABLE_THRESHOLD", 0.7))
        q = float(getattr(cfg, "PHYS_Q", 0.7))
        qt = torch.quantile(resid.flatten(1), q, dim=1).view(b, 1, 1, 1)
        stable_mask = ((stable > threshold) & (resid <= qt)).to(tc.dtype)
        stable_mask[:, :, 0, :] = 0
        stable_mask[:, :, -1, :] = 0
        stable_mask[:, :, :, 0] = 0
        stable_mask[:, :, :, -1] = 0
        valid = valid.reshape(-1).to(tc).clamp(0, 1).view(b, 1, 1, 1)
        valid = valid * roi_valid.to(tc).view(b, 1, 1, 1)
        stable_mask = stable_mask * valid
        return dict(
            forecast=t_forecast,
            cur=tc,
            stable_mask=stable_mask,
            valid=valid,
            bc=bc,
            roi_valid=roi_valid,
            common_fraction=common_fraction,
        )


def physics_loss(mu, phys, ratio=0.0):
    if ratio <= 0.0:
        return mu.sum() * 0.0
    p = crop_roi(mu, phys["bc"], phys["cur"].shape[-1])
    p = masked_norm(p)
    target = masked_norm(phys["forecast"])
    mask = phys["stable_mask"]
    return ((p - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def physics_sigma_nll(sigma, phys, sigma_min=0.15, clip=0.8):


    sr = crop_roi(sigma, phys["bc"], phys["cur"].shape[-1]).clamp_min(float(sigma_min))
    residual = (phys["cur"] - phys["forecast"]).abs().detach().clamp_max(float(clip))
    mask = phys["stable_mask"].detach()
    per_pixel = residual / sr + torch.log(sr / float(sigma_min))
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)
