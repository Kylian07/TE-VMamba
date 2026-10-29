# %% [markdown]
# ## ▶ Run
# Each dataset is an independent training run, as in the paper. Finished datasets are skipped on re-runs
# (`final_results.json`), and unfinished ones resume from `last_state.pth`.

# %%
RESULTS = {}
for name in DATASETS_TO_RUN:
    if name not in PREPARED:
        print(f"[skip] {name}: data not available")
        continue
    r = train_and_eval(name)
    if r is None:
        print("[stop] session time budget reached; re-run with this notebook's output attached to continue.")
        break
    RESULTS[name] = r
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# %% [markdown]
# ## Results
# `PAPER_REPORTED` holds the paper's numbers (CVPR 2026, Table 2: TE-VMamba-Tiny on Market-1501; Table 3: Occluded-REID,
# backbone size not stated). The paper does not evaluate on DukeMTMC-reID or Occluded-Duke, so there is nothing to
# compare those runs with.

# %%
import matplotlib.pyplot as plt

PAPER_REPORTED = {   # (mAP %, Rank-1 %) from the paper; None = not reported in the paper
    "market1501": (84.2, 93.6),   # Table 2, TE-VMamba-Tiny (Small 85.7/95.1, Base 87.1/96.7)
    "occ_reid": (85.3, 95.5),     # Table 3, "TE-VMamba (Ours)", backbone size not stated
    "dukemtmc": None, "occ_duke": None,   # not evaluated in the paper (only in the repo's configs)
}
SHORT = {"market1501": "Market-1501", "dukemtmc": "DukeMTMC", "occ_duke": "Occluded-Duke", "occ_reid": "Occluded-REID"}

rows = []
for name in DATASETS_TO_RUN:
    p = osp.join(OUT_DIR, run_key(name), "final_results.json")
    if osp.isfile(p):
        rows.append(json.load(open(p)))
if not rows:
    print("no finished runs yet")
else:
    print(f"{'dataset':15s}{'mAP':>7s}{'R1':>7s}{'R5':>7s}{'R10':>7s} | {'paper mAP/R1':>13s} | "
          f"{'no-adapt mAP':>12s} | {'tokens':>7s} {'FLOPs~':>7s} {'speed':>7s}")
    for r in rows:
        f, na, e = r["final"], r["token_adapt_off_at_test"], r["efficiency"]
        paper = PAPER_REPORTED.get(r["dataset"])
        ps = f"{paper[0]:.1f}/{paper[1]:.1f}" if paper else "-"
        print(f"{SHORT[r['dataset']]:15s}{100 * f['mAP']:7.1f}{100 * f['R1']:7.1f}{100 * f['R5']:7.1f}{100 * f['R10']:7.1f} | "
              f"{ps:>13s} | {100 * na['mAP']:12.1f} | {-e['token_reduction_%']:6.1f}% {-e.get('flops_proxy_reduction_%', float('nan')):6.1f}% "
              f"x{e['speedup_x']:.2f}")
    print(f"\nsettings: PROTOCOL={PROTOCOL} (batch {rows[0].get('batch', 64)}), MODEL_SIZE={rows[0]['model_size']}, epochs={[r['epochs'] for r in rows]}, "
          f"init: {rows[0]['pretrain']}, RUN_MODE={rows[0]['run_mode']}")
    print("columns: 'no-adapt mAP' = same trained weights with pruning/merging switched off at test; "
          "'tokens' = change in tokens processed per forward; 'FLOPs~' = tokens weighted by block width^2 (compare with the "
          "paper's Table 4: Tiny 5.02 -> 1.86 GFLOPs = -63%); 'speed' = throughput ratio adapt-on / adapt-off")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    names = [SHORT[r["dataset"]] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    axes[0].bar(x - w / 2 - 0.01, [100 * r["final"]["mAP"] for r in rows], w, color="#2a78d6", label="mAP")
    axes[0].bar(x + w / 2 + 0.01, [100 * r["final"]["R1"] for r in rows], w, color="#eb6834", label="Rank-1")
    for xi, r in zip(x, rows):
        axes[0].text(xi - w / 2, 100 * r["final"]["mAP"] + 1, f"{100 * r['final']['mAP']:.1f}", ha="center", fontsize=8, color="#52514e")
        axes[0].text(xi + w / 2, 100 * r["final"]["R1"] + 1, f"{100 * r['final']['R1']:.1f}", ha="center", fontsize=8, color="#52514e")
    axes[0].set_xticks(x, names); axes[0].set_ylim(0, 105); axes[0].set_ylabel("%")
    axes[0].set_title(f"TE-VMamba reproduction ({PROTOCOL} protocol): accuracy", loc="left", fontsize=11)
    axes[0].legend(frameon=False); axes[0].grid(axis="y", color="#e6e5e0"); axes[0].set_axisbelow(True)
    e = rows[0]["efficiency"]
    axes[1].plot(e["off"]["tokens_per_block"], color="#52514e", lw=2, label="pruning/merging off")
    axes[1].plot(e["on"]["tokens_per_block"], color="#2a78d6", lw=2, label="pruning/merging on")
    axes[1].set_xlabel("VSS block index"); axes[1].set_ylabel("tokens entering block"); axes[1].set_yscale("log")
    axes[1].set_title(f"Tokens per block ({names[0]})", loc="left", fontsize=11)
    axes[1].legend(frameon=False); axes[1].grid(color="#e6e5e0")
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    plt.tight_layout(); fig.savefig(osp.join(OUT_DIR, "results.png"), dpi=150); plt.show()

if RUN_MODE == "quick":
    print("\n(quick mode: 2 short epochs + subsampled evaluation, so these numbers are NOT results)")

# %% [markdown]
# ## Outputs (`/kaggle/working/TEVMamba_outputs/`)
# * `<dataset>/final_results.json`: final metrics, the no-adaptation diagnostic, the efficiency measurement and the
#   evaluation history.
# * `<dataset>/train_log.txt` and `<dataset>/final_model.pth`.
# * `<dataset>/last_state.pth`: only while a run is unfinished; needed to resume.
# * `kernel_cache/`: the compiled CUDA kernel, reused when this output is attached to a later session.
# * `results.png`.
