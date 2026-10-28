# %% [markdown]
# ## TE-VMamba backbone (embedded from `model/backbones/`)
# Five files are written verbatim, except for three compatibility patches that are clearly marked in the code
# (`# [notebook patch]`):
# 1. `vmamba.py`: the `fvcore` import (only used by the FLOP counter) and the `mamba2` import (only used by the
#    unused `m0` forward types; its triton files are not embedded) are made optional.
# 2. `csms6s.py`: if the CUDA extension is missing, the scan falls back to VMamba's own chunked PyTorch
#    implementation (`selective_scan_easy` from `kernels/selective_scan/test_selective_scan_easy.py`, chunk 16)
#    instead of the 1-step-at-a-time loop, which is unusably slow at 3,136 tokens.
# 3. `csm_triton.py`: `torch.cuda.device(...)` is skipped for CPU tensors, so the notebook can be smoke-tested on
#    a CPU. Behaviour on GPU is unchanged.
#
# **What these files implement** (see the report for details):
# * `VSSM` / `VSSBlock` / `SS2D` (`v05_noz` forward): VMamba backbone with 4-direction cross-scan + selective scan.
# * `LayerTokenPrune`: token importance = |Δ| (mean SSM step size), threshold θ = μ − α·σ with α = softplus(0.7)
#   ≈ 1.10. It keeps tokens with importance ≥ θ, packs them into a smaller H × ⌈K/H⌉ grid, and later blocks run on
#   fewer tokens.
# * `LayerTokenMerge` + `FixedPairThresholdMerge`: pairs neighbouring tokens (alternating horizontal/vertical), with
#   cosine similarity s and a per-image threshold θ = sigmoid(logit(0.5) + MLP(mean token)). Pairs with s > θ are
#   averaged into one token.

# %%
BB_DIR = osp.join(SRC_DIR, "tevmamba", "backbones")
os.makedirs(BB_DIR, exist_ok=True)
open(osp.join(SRC_DIR, "tevmamba", "__init__.py"), "w").write("")
open(osp.join(BB_DIR, "__init__.py"), "w").write("")
print("backbone sources ->", BB_DIR)

# %% [embed] TEVmamba/model/backbones/AdaptivePairMergeGate.py -> tevmamba/backbones/AdaptivePairMergeGate.py
# %% [embed] TEVmamba/model/backbones/LayerTokenReducer.py -> tevmamba/backbones/LayerTokenReducer.py
# %% [embed] TEVmamba/model/backbones/csms6s.py -> tevmamba/backbones/csms6s.py
# %% [embed] TEVmamba/model/backbones/csm_triton.py -> tevmamba/backbones/csm_triton.py
# %% [embed] TEVmamba/model/backbones/vmamba.py -> tevmamba/backbones/vmamba.py

# %%
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
for k in [k for k in sys.modules if k.startswith("tevmamba")]:
    del sys.modules[k]                       # re-import cleanly if the cell is re-run
from tevmamba.backbones import vmamba as vmamba_mod
from tevmamba.backbones import csms6s
from tevmamba.backbones.vmamba import VSSM
print(f"selective scan backend: {'CUDA oflex kernel' if csms6s.WITH_SELECTIVESCAN_OFLEX else 'chunked PyTorch fallback (slow)'} | "
      f"triton cross-scan: {vmamba_mod.__dict__.get('cross_scan_fn') is not None and __import__('tevmamba.backbones.csm_triton', fromlist=['x']).WITH_TRITON}")
