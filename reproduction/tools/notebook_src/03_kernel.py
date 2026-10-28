# %% [markdown]
# ## Selective-scan CUDA kernel (embedded from `kernels/selective_scan`, compiled here)
# VMamba's SS2D layer scans each feature map along 4 directions with Mamba's *selective scan*. The authors ship a
# CUDA implementation (the `oflex` variant from VMamba, built on Mamba's kernels). The files below are copied
# **verbatim** from the repo, written to disk, and compiled for the GPU in this session (Kaggle T4 = sm_75).
#
# The compiled extension is cached in the notebook output. When that output is attached to a later session, the
# build is skipped.

# %%
KERNEL_DIR = osp.join(SRC_DIR, "selective_scan")
os.makedirs(osp.join(KERNEL_DIR, "csrc", "selective_scan", "cusoflex"), exist_ok=True)
print("kernel sources ->", KERNEL_DIR)

# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/selective_scan.h -> selective_scan/csrc/selective_scan/selective_scan.h
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/selective_scan_common.h -> selective_scan/csrc/selective_scan/selective_scan_common.h
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/static_switch.h -> selective_scan/csrc/selective_scan/static_switch.h
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cub_extra.cuh -> selective_scan/csrc/selective_scan/cub_extra.cuh
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/reverse_scan.cuh -> selective_scan/csrc/selective_scan/reverse_scan.cuh
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/uninitialized_copy.cuh -> selective_scan/csrc/selective_scan/uninitialized_copy.cuh
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_oflex.h -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_oflex.h
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_oflex.cpp -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_oflex.cpp
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_fwd_kernel_oflex.cuh -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_fwd_kernel_oflex.cuh
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_bwd_kernel_oflex.cuh -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_bwd_kernel_oflex.cuh
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu
# %% [embed] TEVmamba/kernels/selective_scan/csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu -> selective_scan/csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu

# %% [markdown]
# ### Build
# This is the same extension, sources and compiler flags as the repo's `setup.py` (`MODES = ["oflex"]`), with three
# build-level adaptations. **None of them changes the kernel source.**
# * The GPU architecture is set explicitly from the detected device (`-arch=sm_XX`).
# * The C++ standard is raised to C++20 when the installed PyTorch headers require it (recent PyTorch refuses
#   C++17). The repo pins c++17 for torch 2.2.
# * A tiny compatibility header is force-included (`-include tev_cub_compat.cuh`). Newer CUDA (CCCL/CUB 3.x) removed
#   `cub::LaneId()` and `cub::CTA_SYNC()`, which `reverse_scan.cuh` calls. The header re-defines them (lane-id
#   register read / `__syncthreads()`) only when CUB ≥ 3.0, so older toolkits are unaffected.

# %%
def _so_tag():
    cc = "".join(map(str, torch.cuda.get_device_capability(0))) if DEVICE == "cuda" else "cpu"
    return f"torch{torch.__version__.split('+')[0]}_cu{torch.version.cuda}_sm{cc}_py{sys.version_info[0]}{sys.version_info[1]}"


def _try_import_kernel(path_dir):
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)
    try:
        import importlib
        mod = importlib.import_module("selective_scan_cuda_oflex")
        return mod
    except Exception as e:  # noqa: BLE001
        print("   import failed:", type(e).__name__, str(e)[:200])
        return None


def build_selective_scan():
    if DEVICE != "cuda":
        print("[kernel] no GPU -> skipping CUDA build (pure-PyTorch scan will be used)")
        return False
    tag = _so_tag()
    cache_dir = osp.join(OUT_DIR, "kernel_cache", tag)
    # 1) cached build from this session or an attached previous output
    cands = glob.glob(osp.join(cache_dir, "selective_scan_cuda_oflex*.so")) + \
        glob.glob(osp.join(INPUT_ROOT, "**", "kernel_cache", tag, "selective_scan_cuda_oflex*.so"), recursive=True)
    for so in cands:
        d = osp.join(CACHE_DIR, "kernel_import"); os.makedirs(d, exist_ok=True)
        shutil.copy(so, d)
        if _try_import_kernel(d):
            print(f"[kernel] using cached build {so}")
            return True
    # 2) compile
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if osp.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc is None:
        print("[kernel] nvcc not found -> cannot compile; pure-PyTorch scan will be used")
        return False
    cc = "".join(map(str, torch.cuda.get_device_capability(0)))
    torch_inc = osp.join(osp.dirname(torch.__file__), "include")
    needs20 = any("C++20 or later" in open(f, errors="ignore").read()
                  for f in (osp.join(torch_inc, "ATen", "ATen.h"), osp.join(torch_inc, "torch", "all.h")) if osp.isfile(f))
    std = "c++20" if needs20 else "c++17"
    compat = osp.join(KERNEL_DIR, "tev_cub_compat.cuh")
    open(compat, "w").write(
        "#pragma once\n#include <cub/version.cuh>\n"
        "#if defined(CUB_VERSION) && CUB_VERSION >= 300000\n"
        "namespace cub {\n"
        "__device__ __forceinline__ unsigned int LaneId() { unsigned int r; asm volatile(\"mov.u32 %0, %%laneid;\" : \"=r\"(r)); return r; }\n"
        "__device__ __forceinline__ void CTA_SYNC() { __syncthreads(); }\n"
        "}\n#endif\n")
    setup_py = f'''
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
this_dir = os.path.dirname(os.path.abspath(__file__))
setup(
    name="selective_scan", version="0.0.2", packages=[],
    ext_modules=[CUDAExtension(
        name="selective_scan_cuda_oflex",
        sources=["csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
                 "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
                 "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu"],
        extra_compile_args={{"cxx": ["-O3", "-std={std}"],
            "nvcc": ["-O3", "-std={std}", "-include", "{compat}", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                     "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-lineinfo",
                     "-arch=sm_{cc}", "--threads", "4"]}},
        include_dirs=[Path(this_dir) / "csrc" / "selective_scan"])],
    cmdclass={{"build_ext": BuildExtension}})
'''
    open(osp.join(KERNEL_DIR, "setup.py"), "w").write(setup_py)
    env = dict(os.environ, MAX_JOBS="4", CUDA_HOME=osp.dirname(osp.dirname(nvcc)))
    env["PATH"] = osp.dirname(nvcc) + ":" + env.get("PATH", "")
    print(f"[kernel] compiling for sm_{cc} with {nvcc}, -std={std} (5-10 min) ...")
    t0 = time.time()
    p = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=KERNEL_DIR, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print("[kernel] BUILD FAILED. Last lines:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        return False
    print(f"[kernel] built in {timedelta(seconds=int(time.time() - t0))}")
    so = glob.glob(osp.join(KERNEL_DIR, "selective_scan_cuda_oflex*.so"))[0]
    os.makedirs(cache_dir, exist_ok=True); shutil.copy(so, cache_dir)
    return _try_import_kernel(KERNEL_DIR) is not None


KERNEL_OK = build_selective_scan()
if not KERNEL_OK and DEVICE == "cuda":
    print("!" * 100 + "\n!! The CUDA selective-scan kernel is NOT available: training will use a chunked pure-PyTorch scan,"
          "\n!! which gives the same maths but is many times slower. Fix the build before a full run.\n" + "!" * 100)
