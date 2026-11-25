from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PRLTOptions:
    layer_index: int = 6
    template_policy: str = "first_template_in_original_order"
    detach_auxiliary_input: bool = True
    sigma_min: float = 0.15
    sigma_max: float = 5.0
    sigma_temperature: float = 1.5
    reliability_beta: float = 2.0
    sigma_prior_weight: float = 0.01
    sigma_smooth_weight: float = 0.01
    sigma_prior_target: float = 0.4
    sigma_mean_target: float = 0.35
    sigma_mean_weight: float = 0.15
    top_anomaly_ratio: float = 0.15
    PHYSICS_REF_MAX_GAP: int = 3
    ROI_SIZE: int = 16
    BOX_SHRINK: float = 0.9
    KAPPA_FIXED: float = 0.08
    PHYS_STABLE_THRESHOLD: float = 0.7
    PHYS_Q: float = 0.7
    ROI_MIN_COMMON_FRACTION: float = 0.5
    PRR_ALPHA_CAP: float = 0.7
    PRR_WARM_EPOCHS: int = 10
    PRR_WEIGHT_MIN: float = 0.5
    PRR_WEIGHT_MAX: float = 2.0
    AUX_WARM: float = 0.7
    AUX_IMAGING: float = 0.001
    AUX_HEAT: float = 0.0005
    AUX_REG_MULT: float = 1.0
    GAUSSIAN_MARGIN_COEF: float = 0.3
    PHYS_RECON_WEIGHT: float = 0.0005
    PHYS_SIGMA_WEIGHT: float = 0.0005
    PHYS_SIGMA_CLIP: float = 0.8
    TARGET_DELTA_MAX: float = 0.1
    TARGET_AUX_WEIGHT: float = 0.015
    TARGET_VIEW_WEIGHT: float = 0.003
    TARGET_START_EPOCH: int = 3
    TARGET_VIEW_INTERVAL: int = 4
    TARGET_MIN_IOU: float = 0.2
    TARGET_STABILITY_TAU: float = 0.08
    TARGET_SUBSET: int = 8
    TARGET_ROI_SIZE: int = 8
    TARGET_LOCAL_RADIUS: int = 1
    TARGET_OUTLIER_MAD: float = 2.0
    TARGET_OUTLIER_ABS: float = 0.03
    TARGET_EASY_WEIGHT: float = 0.25
    TARGET_CHANGE_THRESHOLD: float = 0.007


PRLT = PRLTOptions()


def protocol():
    return asdict(PRLT)
