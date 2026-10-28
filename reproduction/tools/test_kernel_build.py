"""Run the notebook's own build_selective_scan() for sm_75 without a GPU (e.g. with the pip CUDA toolkit).

Usage: CUDA_HOME=<toolkit> python test_kernel_build.py <work_dir>   (work_dir/kbuild must hold the kernel sources
that the notebook writes to SRC_DIR/selective_scan)
"""
import glob
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from datetime import timedelta

import torch

S = sys.argv[1]
C = os.environ.get("CUDA_HOME", "/usr/local/lib/python3.11/dist-packages/nvidia/cu13")
os.environ["CPATH"] = C + "/include/cccl"             # pip CUDA keeps CUB here; system toolkits don't need this
torch.cuda.get_device_capability = lambda *a: (7, 5)  # pretend we are on a T4
nb = json.load(open(osp.join(osp.dirname(osp.abspath(__file__)), "..", "TE-VMamba_Kaggle_Reproduction.ipynb")))
src = next("".join(c["source"]) for c in nb["cells"] if "def build_selective_scan" in "".join(c["source"]))
src = src[:src.index("KERNEL_OK = build_selective_scan()")]
g = dict(os=os, osp=osp, sys=sys, glob=glob, shutil=shutil, subprocess=subprocess, time=time, timedelta=timedelta,
         torch=torch, DEVICE="cuda", OUT_DIR=S + "/kout", INPUT_ROOT=S + "/none", CACHE_DIR=S + "/kcache",
         KERNEL_DIR=S + "/kbuild")
_which = shutil.which
shutil.which = lambda n: C + "/bin/nvcc" if n == "nvcc" else _which(n)
exec(src, g)
t0 = time.time()
ok = g["build_selective_scan"]()
print("BUILD OK:", ok, f"{time.time() - t0:.0f}s")
print("so files:", glob.glob(S + "/kbuild/*.so"), glob.glob(S + "/kout/kernel_cache/*/*.so"))
