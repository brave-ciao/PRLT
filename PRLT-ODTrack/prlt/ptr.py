import torch
from torch import nn
from torch.nn import functional as F

from lib.models.odtrack.vit import _init_vit_weights
from .options import PRLT

PTR_PREFIX = "backbone.prlt_ptr."
TARGET_PREFIX = "prlt_refiner."
PREFIXES = (PTR_PREFIX, TARGET_PREFIX)


class PRLTPTR(nn.Module):
    def __init__(self, cfg, options=PRLT):
        super().__init__()
        self.options = options
        self.search_hw = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.template_hw = cfg.DATA.TEMPLATE.SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.phys_head = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 2))
        self.phys_head.apply(_init_vit_weights)
        self.register_buffer("lap_kernel", torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3), persistent=False)

    @staticmethod
    def normalized(value):
        return (value-value.mean(dim=(2, 3), keepdim=True)) / (value.std(dim=(2, 3), keepdim=True)+1e-6)

    def forward(self, feature, image, template_count, query_count=1):
        o = self.options
        nz, nx = self.template_hw**2, self.search_hw**2
        expected = query_count + template_count*nz + nx
        if tuple(feature.shape[1:]) != (expected, 768):
            raise ValueError("PRLT token layout mismatch: {} vs {} tokens x 768".format(tuple(feature.shape), expected))
        if not o.detach_auxiliary_input or o.template_policy != "first_template_in_original_order":
            raise ValueError("PRLT requires detached PTR inputs and the declared first-template reference")
        z = feature[:, query_count:query_count+nz].detach()
        x = feature[:, -nx:].detach()
        pz = self.phys_head(z).transpose(1, 2).reshape(-1, 2, self.template_hw, self.template_hw)
        px = self.phys_head(x).transpose(1, 2).reshape(-1, 2, self.search_hw, self.search_hw)
        mu, mu_z = px[:, :1], pz[:, :1]
        mu_z_up = F.interpolate(mu_z, size=mu.shape[-2:], mode="bilinear", align_corners=False)
        sigma = o.sigma_min + (o.sigma_max-o.sigma_min)*torch.sigmoid(px[:, 1:]/o.sigma_temperature)
        reliability = torch.exp(-o.reliability_beta*(sigma-o.sigma_min)).clamp(0, 1)
        gray = F.interpolate(image, size=mu.shape[-2:], mode="bilinear", align_corners=False).mean(1, keepdim=True)
        residual = self.normalized(mu)-self.normalized(gray)
        imaging = (residual.abs()/(sigma+1e-6)+torch.log((sigma+1e-6)/o.sigma_min)).mean()
        lap = self.lap_kernel.to(dtype=mu.dtype)
        delta = F.conv2d(mu, lap, padding=1)
        heat = (delta.square()*reliability).sum()/reliability.sum().clamp_min(1.0)
        anomaly = ((mu-mu_z_up).square()+0.5*delta.square()).detach()
        flat = anomaly.flatten(1)
        k = max(1, int(o.top_anomaly_ratio*flat.shape[1]))
        threshold = torch.kthvalue(flat, flat.shape[1]-k+1, dim=1).values[:, None, None, None]
        region = 1-(anomaly>threshold).float()
        prior = ((sigma-o.sigma_prior_target).square()*region).sum()/region.sum().clamp_min(1.0)
        smooth = F.conv2d(sigma, lap, padding=1).square().mean()
        mean_sigma = (sigma*region).sum()/region.sum().clamp_min(1.0)
        regularizer = (o.sigma_prior_weight*prior + o.sigma_smooth_weight*smooth
                       + o.sigma_mean_weight*(mean_sigma-o.sigma_mean_target).abs())
        return {"Tx_mu": mu, "Tx_sigma": sigma, "R_map": reliability,
                "E_map": residual, "loss_imaging": imaging, "loss_heat": heat,
                "loss_phys_reg": regularizer}


def attach_modules(net, cfg, options=PRLT):
    core = getattr(net, "module", net)
    if cfg.MODEL.BACKBONE.CAT_MODE != "direct" or core.token_len != 1 or not core.backbone.add_cls_token:
        raise ValueError("PRLT requires the verified direct ODTrack layout and one temporal token")
    if hasattr(core.backbone, "prlt_ptr") or hasattr(core, "prlt_refiner"):
        if not (hasattr(core.backbone, "prlt_ptr") and hasattr(core, "prlt_refiner")):
            raise RuntimeError("Incomplete PRLT module attachment")
        return core.backbone.prlt_ptr, core.prlt_refiner

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        ptr = PRLTPTR(cfg, options)
        target = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(target[-1].weight)
        nn.init.zeros_(target[-1].bias)
    p = next(core.parameters())
    ptr.to(device=p.device, dtype=p.dtype).train(core.training)
    target.to(device=p.device, dtype=p.dtype).train(core.training)

    core.backbone.add_module("prlt_ptr", ptr)
    core.add_module("prlt_refiner", target)
    return ptr, target


def tracking_state_dict(net):
    core = getattr(net, "module", net)
    return {name: value for name, value in core.state_dict().items() if not name.startswith(PREFIXES)}


def auxiliary_state_dict(net):
    core = getattr(net, "module", net)
    if not hasattr(core.backbone, "prlt_ptr"):
        return {}
    return {"ptr": core.backbone.prlt_ptr.state_dict(), "target_refiner": core.prlt_refiner.state_dict()}
