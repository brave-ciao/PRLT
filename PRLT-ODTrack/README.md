# PRLT-ODTrack

Implementation of **Physics-Inspired Reliability Learning for Robust Thermal Infrared Object Tracking** with the ODTrack-B backbone. Physics-Inspired Thermal Reconstruction (PTR), Uncertainty-Aware Reliability Modeling (URM), and Progressive Capped Reliability Reweighting (PRR) are used during training. Inference uses the ODTrack backbone, temporal tokens, and tracking head.

🚀 **Test Results and Model**

[Test Results](https://pan.baidu.com/s/1WdlHER9aRDvi5YWasD93aA) [Trained Model](https://pan.baidu.com/s/14aMNAEx-5-XR9ENmrpkCHw)  (Baidu Netdisk, access code `PRLT`)

## Installation

The reference environment uses Python 3.10.4 on Linux x86-64 with PyTorch `2.14.0+cu130` and torchvision `0.29.0+cu130`. GPU training and evaluation require an NVIDIA driver compatible with CUDA 13.0. Install the [matching PyTorch wheels](https://pytorch.org/get-started/previous-versions/) before installing the remaining dependencies.

```bash
conda create -n prlt python=3.10.4 pip -y
conda activate prlt
python -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

## Data and checkpoints

Set the local paths in [local_settings.py](local_settings.py).

| Split | Path setting | Expected layout |
| --- | --- | --- |
| LSOTB-TIR training | `LSOTB_TRAIN_DIR` | One directory per sequence with images and `groundtruth.txt` or `groundtruth_rect.txt`. If provided, `list.txt` at the split root specifies the sequences. |
| LSOTB-TIR120 | `LSOTB120_ROOT` | Official 120 test sequences with dense `xywh` annotations. |
| PTB-TIR | `PTBTIR_ROOT` | Official 60 test sequences. Set `PTBTIR_ANNOTATION_ROOT` to the official [PTB-TIR toolkit](https://github.com/QiaoLiuHit/PTB-TIR_Evaluation_toolkit) annotation directory if the ground truth is stored separately. |
| VTUAV-TIR ST | `VTUAVTIR_ROOT` or `VTUAVTIR_ST_ROOT` | Official [short-term test sequences](https://github.com/zhang-pengyu/DUT-VTUAV). Each sequence uses full-rate `ir/` frames and the provided `ir.txt` and `rgb.txt` annotations. |
| LasHeR-TIR | `LASHERTIR_ROOT` | Official 245 test sequences with `infrared/` images and one `xywh` row per IR frame in `init.txt`. |

The LSOTB-TIR training reader expects `xyxy` annotations by default. Set `LSOTB_TRAIN_BOX_FORMAT = "xywh"` for an archive with width and height annotations. Use the official dataset distributions for images and annotations.

For training, download the ODTrack-B **Base-Fulldata-300ep** checkpoint (`ODTrack_ep0300.pth.tar`) from the [ODTrack model collection](https://drive.google.com/drive/folders/17LacrfRO01R75bxU4bgA87eo1b_rX5Gj?usp=sharing). Place it in `pretrained_networks/` or set `PRETRAINED_TRACKER` to its location. Use a trained PRLT-ODTrack checkpoint to evaluate PRLT.

## Training

```bash
python train.py
```

The [training configuration](experiments/odtrack/prlt_b384.yaml) uses 192 × 192 templates, 384 × 384 search images, a ViT-B/16 backbone with candidate elimination disabled, and 25 epochs on a single GPU. Checkpoints are saved under `output/checkpoints/train/odtrack/prlt_b384/` by default. Use `python train.py --resume` to continue a run created with this code.

## Evaluation

```bash
python test.py --checkpoint /path/to/ODTrack_ep0025.pth.tar
```

To evaluate one benchmark, add `--dataset lsotb120`, `--dataset ptbtir`, `--dataset vtuavtir`, or `--dataset lashertir`. With the paths and `TEST_CHECKPOINT` set in `local_settings.py`, `train.py` and `test.py` can also be run directly from an IDE. The evaluator writes predictions, per-sequence results, and `summary.csv` under `results/`.

VTUAV tracking uses infrared images. Its MSR and MPR use both annotation streams for scoring. The NPR column uses NP51 on LSOTB-TIR120 and normalized precision at 0.20 on LasHeR-TIR.

## Acknowledgments

This implementation builds on [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack). The upstream copyright notice and license are included in [LICENSE](LICENSE).
