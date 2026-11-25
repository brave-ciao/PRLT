from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

from .focal import FocalLoss
from .physics import compute_physics_maps, physics_loss, physics_sigma_nll
from .temporal import temporal_delta_loss, calibrated_r_inside_box
from .options import PRLT


class FrameFocal(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original, self.weighted = original, FocalLoss()
        self.weights, self.index = None, 0

    @contextmanager
    def bind(self, weights):
        if self.weights is not None:
            raise RuntimeError("Nested PRLT loss calls")
        self.weights, self.index = weights, 0
        try:
            yield
            if self.index != len(weights):
                raise RuntimeError("PRLT weights did not match all ODTrack search frames")
        finally:
            self.weights, self.index = None, 0

    def forward(self, prediction, target):
        if self.weights is None:
            return self.original(prediction, target)
        weight = self.weights[self.index]
        self.index += 1
        return self.weighted(prediction, target, weight=weight)


def reliability_weight(reliability, epoch, options=PRLT):
    progress = (int(epoch) - 1) / float(max(1, options.PRR_WARM_EPOCHS))
    alpha = min(options.PRR_ALPHA_CAP, max(0.0, progress))
    r = reliability.clamp(0, 1)
    r_half = 0.5 * r + 0.5 * r.detach()
    target = 0.2 + 0.8 * r_half.square()
    target = target / (target.mean(dim=(2, 3), keepdim=True) + 1e-6)
    target = target.clamp(options.PRR_WEIGHT_MIN, options.PRR_WEIGHT_MAX)
    return 1.0 - alpha + alpha * target, alpha


def frame_objectives(out, data, index, gaussian, cfg, options=PRLT):
    o = options
    box = data["search_anno"][index]
    base_r, sigma = out["R_base"], out["Tx_sigma"]
    corrected, applied, _ = calibrated_r_inside_box(base_r, out["R_delta"], box, detach_delta_for_prr=True)
    weight, alpha = reliability_weight(corrected, data.get("epoch", 1), o)
    zero = sigma.sum()*0.0
    physics = None
    if "physics_images" in data:
        physics = compute_physics_maps(data["physics_images"][index], data["search_images"][index],
            data["physics_anno"][index].to(sigma), box.to(sigma),
            data["physics_delta"][index], data["physics_valid"][index], o,
            cfg.DATA.MEAN, cfg.DATA.STD)
    gn = gaussian.detach()/(gaussian.detach().amax(dim=(2, 3), keepdim=True)+1e-6)
    pos, neg = (gn>0.3).float(), (gn<0.05).float()
    def wmean(value, mask):
        return (value*mask).sum(dim=(2, 3))/mask.sum(dim=(2, 3)).clamp_min(1.0)
    sp, sn = wmean(sigma, pos), wmean(sigma, neg)
    rp, rn = wmean(base_r, pos), wmean(base_r, neg)
    sigma_gt = (0.03*F.softplus((0.05+sp-sn)/0.03)).mean()
    r_gt = (0.03*F.softplus((0.08-(rp-rn))/0.03)).mean()
    phy_mu = physics_loss(out["Tx_mu"], physics, 1.) if physics else zero
    phy_sigma = physics_sigma_nll(sigma, physics, clip=o.PHYS_SIGMA_CLIP) if physics else zero
    delta = out["R_delta"]
    target, mask = out.get("temporal_teacher"), out.get("temporal_mask")
    stability = delta.square().mean()
    view = delta.sum()*0.0
    risky = 0.0
    if target is not None and mask is not None and bool(mask.any()):
        view = temporal_delta_loss(delta, target, mask, easy_weight=o.TARGET_EASY_WEIGHT,
                                   change_threshold=o.TARGET_CHANGE_THRESHOLD)
        changed = target.abs()>o.TARGET_CHANGE_THRESHOLD
        risky = float((changed*mask.bool()).sum().item()/mask.sum().item())
    auxiliary = (o.AUX_WARM*(o.AUX_IMAGING*out["loss_imaging"] + o.AUX_HEAT*out["loss_heat"]
                 + o.AUX_REG_MULT*out["loss_phys_reg"] + o.GAUSSIAN_MARGIN_COEF*(sigma_gt+r_gt))
                 + o.PHYS_RECON_WEIGHT*phy_mu + o.PHYS_SIGMA_WEIGHT*phy_sigma
                 + o.TARGET_AUX_WEIGHT*view + o.TARGET_VIEW_WEIGHT*stability)
    def number(value):
        return float(value.detach()) if torch.is_tensor(value) else float(value)
    stats = {"physical_contribution": o.PHYS_RECON_WEIGHT*phy_mu+o.PHYS_SIGMA_WEIGHT*phy_sigma,
             "urm_margin_contribution": o.AUX_WARM*o.GAUSSIAN_MARGIN_COEF*(sigma_gt+r_gt),
             "urm_temporal_contribution": o.TARGET_AUX_WEIGHT*view+o.TARGET_VIEW_WEIGHT*stability,
             "weight_source_mean": corrected.detach().mean(),
             "aux": auxiliary, "imaging": out["loss_imaging"], "heat": out["loss_heat"],
             "regularizer": out["loss_phys_reg"], "sigma_gt": sigma_gt, "R_gt": r_gt,
             "physical_mu": phy_mu, "physical_sigma": phy_sigma, "view_target": view,
             "view_stability": stability, "temporal_valid_fraction": out.get("temporal_good_frac", 0.),
             "temporal_risky_fraction": risky, "teacher_executed": out.get("teacher_executed", False),
             "alpha": alpha, "R_mean": corrected.detach().mean(), "R_base_mean": base_r.detach().mean(),
             "delta_abs_mean": applied.detach().abs().mean(), "sigma_mean": sigma.detach().mean(),
             "weight_min": weight.detach().min(), "weight_max": weight.detach().max(),
             "physics_valid": physics["valid"].mean() if physics else 0.,
             "physics_roi_valid": physics["roi_valid"].float().mean() if physics else 0.,
             "physics_common_fraction": physics["common_fraction"].mean() if physics else 0.}
    return weight, auxiliary, {key: number(value) for key, value in stats.items()}
