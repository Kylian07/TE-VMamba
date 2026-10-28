# %% [markdown]
# # TE-VMamba reproduction on Kaggle: SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-ID (CVPR 2026)
#
# This notebook runs the **official TE-VMamba code** (github.com/Kylian07/TE-VMamba) in a single Kaggle notebook,
# with **nothing cloned from GitHub**:
# * The authors' backbone files (`vmamba.py`, `LayerTokenReducer.py`, `AdaptivePairMergeGate.py`, `csms6s.py`,
#   `csm_triton.py`) are **embedded verbatim** in the cells below. Only three small compatibility patches are
#   applied, each listed where it happens.
# * The authors' **CUDA selective-scan kernel** (`kernels/selective_scan`, about 85 KB of source) is embedded and
#   **compiled on Kaggle's GPU** (5–10 min). If the build fails, a slower pure-PyTorch scan is used and the log says
#   so loudly.
# * Data loading, sampler, losses, optimiser, LR schedule, training loop and evaluation are ported 1:1 from
#   `datasets/`, `loss/`, `solver/`, `processor/` and `utils/metrics.py`.
#
# **Protocol (the repo's `v2v_tiny` configs):** VMamba-Tiny (dims 96, depths 2-2-8-2) with ImageNet initialisation,
# 224×224 input, PK sampler (16 identities × 4 images), cross-entropy plus triplet loss (margin 0.3), AdamW
# (lr 3e-4, wd 0.05), cosine schedule with 5 warm-up epochs, AMP, 300 epochs (600 for Occluded-Duke). Token pruning
# runs at blocks 5–7 and pairwise token merging at blocks 9–11.
#
# **Datasets:** Market-1501, DukeMTMC-reID and Occluded-Duke from Kaggle, plus Occluded-REID from its authors'
# official zip. The next cell lists them.
#
# **How to run**
# 1. *Settings → Accelerator → GPU T4 x2* (one GPU is used) and *Internet → On* (needed for ImageNet weights, the
#    Occluded-REID zip and the Occluded-Duke split lists).
# 2. *Add Input* the Kaggle datasets in the next cell.
# 3. Smoke test with `RUN_MODE = "quick"`, then `RUN_MODE = "full"` and **Save Version → Save & Run All**.
# 4. The full protocol is long (see the time estimate printed after the first epoch). The notebook saves a resumable
#    checkpoint every few epochs and stops cleanly before Kaggle's 12 h limit. Attach this notebook's output as an
#    input and run again to continue.

# %% [markdown]
# ## Datasets
#
# | Dataset | Source | Notes |
# |---|---|---|
# | Market-1501 | Kaggle [`pengcw1/market-1501`](https://www.kaggle.com/datasets/pengcw1/market-1501) | standard layout (used successfully in the PAD runs) |
# | DukeMTMC-reID | Kaggle [`whurobin/dukemtmcreid`](https://www.kaggle.com/datasets/whurobin/dukemtmcreid) | standard layout (used successfully in the PAD runs) |
# | Occluded-Duke | **built in the notebook** from `whurobin/dukemtmcreid` and the official Occluded-DukeMTMC name lists (Miao et al., ICCV 2019; `train/query/gallery.list`, downloaded from github.com/lightas/Occluded-DukeMTMC-Dataset) | 15,618 train, 2,210 query, 17,661 gallery |
# | Occluded-REID | **downloaded** from its authors' release (Zhuo et al., ICME 2018; `Occluded_REID.zip`, 42 MB, sha256-checked) and converted | 200 IDs × (5 occluded + 5 full-body). Standard supervised split: a random half of the IDs (seed 0) for training, the rest for testing (occluded images = query, full-body = gallery) |
#
# MSMT17 is not used: its images are not on Kaggle, and the repo's MSMT17 tiny config even points to Market. The
# repo has no CUHK03 loader or config.

# %%
# ---- first code cell: pin ONE GPU (no multi-GPU code path in the repo except DDP) ----
import os, time
T_NOTEBOOK_START = time.time()     # used to stop cleanly before Kaggle's session limit
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("TEV_GPU", "0")

import copy, csv, glob, hashlib, json, logging, math, random, re, shutil, subprocess, sys, time, urllib.request, warnings, zipfile
import os.path as osp
from collections import defaultdict
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as T

try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm
from timm.data.random_erasing import RandomErasing

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="timm")
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | timm {timm.__version__} | device {DEVICE} "
      f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''} | GPUs visible: {torch.cuda.device_count()}")

# %% [markdown]
# ## Run configuration (edit this cell)

# %%
RUN_MODE = os.environ.get("TEV_RUN_MODE", "quick")   # "quick" = smoke test (minutes) | "full" = paper protocol

# Datasets to train+evaluate, in this order (each is an independent run, as in the paper):
DATASETS_TO_RUN = os.environ.get("TEV_DATASETS", "market1501,occ_duke,occ_reid,dukemtmc").split(",")

MODEL_SIZE = "tiny"         # "tiny" (default; official ImageNet weights exist) | "small" | "base" (repo's v2v_* configs)
PRETRAIN = os.environ.get("TEV_PRETRAIN", "imagenet")   # "imagenet": official VMamba ImageNet-1K weights when they match MODEL_SIZE | "none": as shipped (empty path)
EPOCH_SCALE = 1.0           # 1.0 = the repo's schedule (300 / 600 epochs). e.g. 0.5 halves every schedule (state it when reporting!)
EVAL_EVERY = 50             # extra mid-training evaluations every N epochs (the repo only evaluates at the end); 0 = off
CKPT_EVERY = 5              # save a resumable training state every N epochs
MAX_SESSION_HOURS = float(os.environ.get("TEV_MAX_HOURS", "11.3"))   # stop cleanly before Kaggle's 12 h limit
RESUME = True               # continue from the latest state found in OUT_DIR or in an attached previous output

# Paths (Kaggle defaults; env overrides for local testing)
INPUT_ROOT = os.environ.get("TEV_INPUT_ROOT", "/kaggle/input")
WORK_DIR = os.environ.get("TEV_WORK_DIR", "/kaggle/working")
_tmp = "/kaggle/temp" if osp.isdir("/kaggle/temp") else "/tmp"
DATA_ROOT = os.environ.get("TEV_DATA_ROOT", osp.join(_tmp, "tev_data"))
SRC_DIR = os.environ.get("TEV_SRC_DIR", osp.join(_tmp, "tev_src"))         # embedded source files are written here
OUT_DIR = osp.join(WORK_DIR, "TEVMamba_outputs")
CACHE_DIR = os.environ.get("TEV_CACHE", osp.join(_tmp, "tev_cache"))
for d in (DATA_ROOT, SRC_DIR, OUT_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)
DATA_OVERRIDES = {}         # e.g. {"market1501": "/kaggle/input/.../Market-1501-v15.09.15"}
NUM_WORKERS = min(4, os.cpu_count() or 2)

QUICK = dict(epochs=2, max_iters=10, eval_max_query=200, eval_max_gallery=2000)
TEST_IMG_SIZE = os.environ.get("TEV_TEST_SIZE")          # local CPU tests only: e.g. "64" to shrink inputs

if RUN_MODE != "full":
    print("!" * 100 + "\n!! RUN_MODE = \"quick\": SMOKE TEST ONLY. Numbers are NOT results. "
          "Set RUN_MODE = \"full\" for the real experiment.\n" + "!" * 100)
print(f"RUN_MODE={RUN_MODE} | MODEL_SIZE={MODEL_SIZE} | PRETRAIN={PRETRAIN} | EPOCH_SCALE={EPOCH_SCALE} | "
      f"datasets={DATASETS_TO_RUN}\nDATA_ROOT={DATA_ROOT} | OUT_DIR={OUT_DIR}")
