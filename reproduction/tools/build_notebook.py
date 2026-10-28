"""Build reproduction/TE-VMamba_Kaggle_Reproduction.ipynb from tools/notebook_src/*.py.

Cells are separated by `# %%` / `# %% [markdown]` markers. A line
    # %% [embed] <path in repo> -> <path under SRC_DIR>
becomes a `%%writefile` cell holding that repository file verbatim, except for the patches listed in PATCHES
(each must match exactly once, so a changed upstream file fails loudly instead of being silently mis-patched).

Usage (from the repository root):  python reproduction/tools/build_notebook.py
"""
import os
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "notebook_src")
OUT = os.path.join(REPO, "reproduction", "TE-VMamba_Kaggle_Reproduction.ipynb")

_EASY_FALLBACK = '''

# [notebook patch] chunked pure-PyTorch selective scan, used only when no CUDA extension could be imported.
# Copied from kernels/selective_scan/test_selective_scan_easy.py::selective_scan_easy (VMamba authors); code unchanged, comments removed.
def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
    def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
        ts = dts.cumsum(dim=0)
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
        scale = 1
        rAts = Ats / scale
        duts = dts * us
        dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)
        hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0)
        hs = hs_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs)
        return ys, hs

    dtype = torch.float32
    inp_dtype = us.dtype
    has_D = Ds is not None
    if chunksize < 1:
        chunksize = Bs.shape[-1]
    dts = dts.to(dtype)
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).to(dtype)
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    As = As.view(G, -1, N).to(dtype)
    Bs = Bs.permute(3, 0, 1, 2).to(dtype)
    Cs = Cs.permute(3, 0, 1, 2).to(dtype)
    Ds = Ds.view(G, -1).to(dtype) if has_D else None
    D = As.shape[1]
    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=dtype)
    for i in range(0, L, chunksize):
        ys, hs = selective_scan_chunk(us[i:i + chunksize], dts[i:i + chunksize], As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix)
        oys.append(ys)
        hprefix = hs[-1]
    oys = torch.cat(oys, dim=0)
    if has_D:
        oys = oys + Ds * us
    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    return oys.to(inp_dtype) if not return_last_state else (oys.to(inp_dtype), hprefix.view(B, G * D, N).float())


def selective_scan_easy_fallback(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, oflex=True):
    out = selective_scan_easy(u, delta, A, B, C, D, delta_bias, delta_softplus, chunksize=16)
    return out.float() if oflex else out   # oflex=True: the CUDA kernel returns fp32
'''

PATCHES = {
    "TEVmamba/model/backbones/vmamba.py": [
        ("from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count\n",
         "try:  # [notebook patch] fvcore is only used by the FLOP counter\n"
         "    from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count\n"
         "except ImportError:\n"
         "    FlopCountAnalysis = flop_count_str = flop_count = parameter_count = None\n"),
        ("try:\n    from .mamba2.ssd_minimal import selective_scan_chunk_fn\nexcept:\n    from mamba2.ssd_minimal import selective_scan_chunk_fn\n",
         "try:  # [notebook patch] mamba2 (triton) is only used by the unused 'm0' forward types; not embedded\n"
         "    from .mamba2.ssd_minimal import selective_scan_chunk_fn\n"
         "except Exception:\n"
         "    selective_scan_chunk_fn = None\n"),
    ],
    "TEVmamba/model/backbones/csms6s.py": [
        ('    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply\n',
         '    if backend != "torch" and not WITH_CUDA:  # [notebook patch] fast chunked fallback instead of the per-step loop\n'
         '        return selective_scan_easy_fallback(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)\n'
         '    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply\n'),
        ("# fvcore flops =======================================\n",
         _EASY_FALLBACK.lstrip("\n") + "\n\n# fvcore flops =======================================\n"),
    ],
    "TEVmamba/model/backbones/csm_triton.py": [
        ("import torch\nimport warnings\n", "import torch\nimport warnings\nimport contextlib  # [notebook patch]\n"),
        ("    with torch.cuda.device(x.device):\n        return CSF.apply(",
         "    with (torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext()):  # [notebook patch] CPU-safe\n        return CSF.apply("),
        ("    with torch.cuda.device(y.device):\n        return CMF.apply(",
         "    with (torch.cuda.device(y.device) if y.is_cuda else contextlib.nullcontext()):  # [notebook patch] CPU-safe\n        return CMF.apply("),
    ],
}


def embed_cell(src_rel, dest_rel):
    path = os.path.join(REPO, src_rel)
    text = open(path, encoding="utf-8").read()
    for old, new in PATCHES.get(src_rel, []):
        n = text.count(old)
        if n != 1:
            sys.exit(f"patch for {src_rel} matched {n} times (expected 1):\n{old[:120]}")
        text = text.replace(old, new)
    return new_code_cell(f"%%writefile {{SRC_DIR}}/{dest_rel}\n" + text.rstrip("\n") + "\n")


def main():
    cells = []
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".py"):
            continue
        parts = re.split(r"^# %%(.*)$", open(os.path.join(SRC, f), encoding="utf-8").read(), flags=re.M)
        for marker, body in zip(parts[1::2], parts[2::2]):
            m = re.match(r"\s*\[embed\]\s+(\S+)\s+->\s+(\S+)", marker)
            if m:
                cells.append(embed_cell(m.group(1), m.group(2)))
                continue
            body = body.strip("\n")
            if not body.strip():
                continue
            if "[markdown]" in marker:
                cells.append(new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))))
            else:
                cells.append(new_code_cell(body))
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    print(f"{len(cells)} cells -> {OUT}")


if __name__ == "__main__":
    main()
