from pathlib import Path
from .environment import EnvSettings
import local_settings as local


def local_env_settings():
    settings = EnvSettings()
    output = Path(local.OUTPUT_DIR).expanduser().resolve()
    settings.prj_dir = str(local.PROJECT_ROOT)
    settings.save_dir = str(output)
    settings.lsotb_path = str(local.LSOTB120_ROOT)
    settings.results_path = str(output / "test/tracking_results")
    settings.segmentation_path = str(output / "test/segmentation_results")
    settings.result_plot_path = str(output / "test/result_plots")
    settings.network_path = str(local.PROJECT_ROOT / "pretrained_networks")
    return settings
