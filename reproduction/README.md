# TE-VMamba reproduction kit

| File | What it is |
|---|---|
| [`TE-VMamba_Kaggle_Reproduction.ipynb`](TE-VMamba_Kaggle_Reproduction.ipynb) | **Self-contained Kaggle notebook.** Embeds the authors' backbone (`model/backbones/*`) and CUDA selective-scan kernel, compiles the kernel on Kaggle's GPU, prepares Market-1501, DukeMTMC-reID, Occluded-Duke and Occluded-REID, and trains/evaluates with the repo's `v2v_tiny` protocol. Resumes across 12 h sessions. |
| [`REPORT.md`](REPORT.md) | Plain-language methodology report: VMamba background, Δ-based token pruning, pairwise token merging, training protocol, datasets, and **code-vs-paper observations** (verified by running the code). |
| `tools/notebook_src/`, `tools/build_notebook.py` | Notebook source cells, and the builder that embeds the repository files verbatim (with the listed patches). |
| `tools/test_kernel_build.py` | Runs the notebook's kernel build for `sm_75` without a GPU (toolchain check). |
| `tools/run_notebook.py` | Executes the notebook headlessly (used for the CPU smoke test). |

**Quick start on Kaggle:** upload the notebook, choose *GPU T4 x2* with *Internet on*, attach
`pengcw1/market-1501` and `whurobin/dukemtmcreid` (Occluded-Duke and Occluded-REID are built and downloaded by the
notebook), run once with `RUN_MODE = "quick"`, then set `RUN_MODE = "full"` and use *Save & Run All*. Read the time
estimate printed after the first epoch. The full protocol for all four datasets needs more than one week of
Kaggle GPU quota (REPORT §7).
