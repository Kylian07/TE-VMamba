# TE-VMamba: SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-ID
### A plain-language methodology report and reproduction guide

> **Paper:** Huiyuan Huang, Sang Min Yoon. *SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging
> for Person Re-Identification.* CVPR 2026 (poster). Code: this repository (`TEVmamba/`).
>
> **Companion notebook:** [`TE-VMamba_Kaggle_Reproduction.ipynb`](TE-VMamba_Kaggle_Reproduction.ipynb) is a single
> Kaggle notebook that embeds the authors' backbone and CUDA kernel, prepares the datasets from Kaggle and official
> sources, and trains and evaluates with the repository's protocol.
>
> **How this report was made:** from a line-by-line reading of the code in this repository, checked by running it
> (§6), and from the paper PDF (CVPR 2026 Open Access version, pp. 4363–4372). §9 compares the paper, the
> repository and our Kaggle run side by side.

---

## 1. The problem in one paragraph

**Person re-identification (ReID):** given a photo of a person from one camera, find the same person among
thousands of photos from other cameras. A network turns each photo into a feature vector, and the gallery is ranked
by similarity. State-of-the-art ReID backbones (ViTs such as TransReID) are accurate but expensive. **VMamba**, a
vision version of the Mamba state-space model, has *linear* rather than quadratic cost in the number of image
patches ("tokens"). It still processes **every** token in every layer, though, even background patches that say
nothing about identity. TE-VMamba asks: *can we drop or combine unimportant tokens inside VMamba, and save compute
without losing accuracy?* The abstract reports **over 60% fewer FLOPs on Market-1501 with competitive accuracy**.

## 2. Background: VMamba in five sentences

1. The image (224×224) is cut into patches by a small conv "stem", giving a 56×56 grid of tokens. Four stages then
   halve the grid each time (56 → 28 → 14 → 7) while widening the channels (96 → 192 → 384 → 768 for the Tiny model).
2. Each stage is a stack of **VSS blocks**. Each block has an **SS2D** layer (the "attention replacement") followed by
   an MLP, both with residual connections.
3. **SS2D** flattens the 2D token grid into 1D sequences along **4 scan directions** (row-wise, column-wise, and both
   reversed) and runs Mamba's **selective scan** on each. This is a recurrent state update
   *h_t = exp(Δ_t·A)·h_{t−1} + Δ_t·B_t·x_t*, *y_t = C_t·h_t + D·x_t*. The 4 outputs are then merged back into a grid.
4. **Δ (delta, the "step size")** is computed from the token itself. A large Δ means the token strongly *writes into*
   the state (it matters); a small Δ means it barely changes it. **TE-VMamba uses Δ as its measure of token
   importance**, which is what "SSM-aware" means.
5. The selective scan runs as a custom CUDA kernel (`kernels/selective_scan`), so the notebook has to compile it.

## 3. TE-VMamba's two additions

```
blocks:  0 1 | 2 3 | 4  5  6  7  8  9 10 11 | 12 13        (VMamba-Tiny: depths 2-2-8-2 = 14 blocks)
                         └prune┘   └merge─┘                 (repo's tiny config: prune 5,6,7; merge 9,10,11)
grid:    56² | 28² |            14²         |  7²
```

### 3.1 Layer-adaptive token pruning (shallow layers) · `LayerTokenReducer.py::LayerTokenPrune`
Inside SS2D of blocks 5–7:
1. **Importance per token** = mean |Δ| over all channels and the 4 scan directions (`delta_map` in
   `vmamba.py::forward_corev2`).
2. **Adaptive threshold per image**: θ = μ − α·σ, where μ and σ are the mean and standard deviation of the
   importances in that image. α = softplus(α_raw) is meant to be a *learnable* layer-wise parameter.
3. Keep tokens with importance ≥ θ (at least 5% of tokens are always kept).
4. **Pack** the kept tokens, in raster order, into a smaller grid of height H and width ⌈K/H⌉ (padding with zeros).
   The block's residual input is resized (average-pooled or bilinearly interpolated) to that new grid, and **all later
   blocks run on fewer tokens**. This is where the compute saving comes from.

With α = softplus(0.7) ≈ 1.10, roughly the 86% of tokens whose importance is above "mean − 1.1 std" are kept per
pruning layer (if importances were normally distributed). Three pruning layers compound this.

### 3.2 Depth-aware pairwise token merging (deep layers) · `AdaptivePairMergeGate.py::FixedPairThresholdMerge`
Inside SS2D of blocks 9–11:
1. **Fixed pairs** of neighbouring tokens: (0,1), (2,3), … along rows on even layers and along columns on odd
   layers. That is O(P) comparisons, not the all-pairs matching of ToMe.
2. **Similarity** s = cosine(a, b) for each pair.
3. **Per-image threshold** θ = sigmoid(θ_base + MLP(mean token)), a small 2-layer MLP gate (hidden 64), with θ_base
   initialised so that sigmoid(θ_base) = 0.5.
4. Pairs with s > θ are **averaged into one token**. Others stay as two. The result is re-packed into a grid, as for
   pruning.

### 3.3 Everything else is a standard strong ReID baseline (TransReID-style)
Global average pooling → BNNeck → linear ID classifier. Loss = cross-entropy (ID) + batch-hard triplet loss
(margin 0.3) on the pre-BNNeck feature. Test feature = pooled feature before BNNeck, L2-normalised, cosine distance.

## 4. Training protocol (repository configs, `configs/<dataset>/v2v_tiny.yml`)

| Setting | Value |
|---|---|
| Backbone | VMamba-Tiny `s1l8`: dims 96, depths 2-2-8-2, d_state 1, ssm_ratio 1.0, forward `v05_noz`, drop-path 0.2 |
| Token adaptation | prune blocks 5,6,7 · merge blocks 9,10,11 (merge allowed from block 8) |
| Input | 224×224, flip 0.5, pad 10 + random crop, random erasing 0.3, ImageNet mean/std |
| Sampler / batch | PK: 16 identities × 4 images = 64 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 (0 for biases/norms) |
| Schedule | cosine over all epochs, 5 warm-up epochs from 1% lr, floor 0.2% lr, AMP fp16 |
| Epochs | 300 (Market-1501, DukeMTMC, Occluded-REID), 600 (Occluded-Duke, MSMT17) |
| Evaluation | once, after the last epoch (`EVAL_PERIOD = MAX_EPOCHS`); mAP and CMC; no re-ranking |
| Seed | 1234 |

Base and Small configs differ only in width/depth (dims 128 / 96, depths 2-2-20-2), drop-path (0.5 / 0.3) and the
merge layers (12–16).

## 5. Datasets used in the notebook

| Dataset | How obtained | Split |
|---|---|---|
| Market-1501 | Kaggle `pengcw1/market-1501` | standard 751 / 750 IDs |
| DukeMTMC-reID | Kaggle `whurobin/dukemtmcreid` | standard 702 / 1,110 IDs |
| Occluded-Duke | built in the notebook from `whurobin/dukemtmcreid` + the official name lists of Miao et al. (ICCV 2019) | 15,618 train / 2,210 query / 17,661 gallery |
| Occluded-REID | official `Occluded_REID.zip` from Zhuo et al. (ICME 2018), sha256-checked | 200 IDs. The repo expects a pre-split `train/` and `test/{query,gallery}` folder but does not ship the split, so the notebook uses the common supervised protocol: a random half of the IDs (seed 0) for training, occluded images as query, full-body images as gallery |

Not used: **MSMT17** (images not on Kaggle; also, the repo's MSMT17 tiny config points to Market by mistake) and
**CUHK03** (mentioned in the README, but the repo has no CUHK03 loader or config).

## 6. Implementation observations (code vs. the paper's description)

These were found by reading the code and **confirmed by running it** (gradient check in the notebook's test
environment). They matter for interpreting any reproduced number, and are worth raising with your guide. The
notebook keeps the official behaviour in every case, so its results are comparable to the released code.

1. **The "learnable thresholds" never learn.** `vmamba.py` calls the prune and merge modules inside
   `torch.no_grad()` (`_apply_prune_once`, `_apply_merge_once`). As a result, α_raw (pruning), θ_base and the
   gate MLP (merging) **receive no gradient** and stay at their initial values for the whole training: α = 1.10, and
   θ = sigmoid(0 + randomly initialised MLP). The abstract's "learnable layer-wise thresholds" are fixed in the
   released code.
2. **The SSM core of the 6 prune/merge blocks is frozen as a side effect.** Because the SS2D output passes through
   that `no_grad` call, the gradient stops there. In blocks 5–7 and 9–11, `in_proj`, the depth-wise conv, `x_proj`,
   the Δ projection, A and D get **no gradient**; only `out_proj` and the MLP train (verified: `in_proj.grad` and
   `A_logs.grad` are `None` in exactly those blocks). With ImageNet initialisation these layers keep their ImageNet
   values. With the configs as shipped (random init, see 4), they stay **random** forever.
3. **Pruning scrambles the 2-D layout.** Kept tokens are packed row-major into an H × ⌈K/H⌉ grid, so later 2-D
   cross-scans no longer see true spatial neighbours. The residual branch is resized by pooling/interpolation, not
   gathered with the same keep-mask, so the residual and the SS2D output no longer line up token-for-token.
4. **No ImageNet weights are loaded as shipped.** Every `v2v_*.yml` has `PRETRAIN_CHOICE: imagenet` but
   `PRETRAIN_PATH: ""`, and the loader only runs when the path is non-empty. Official VMamba ImageNet checkpoints
   exist for Tiny `s1l8` (an exact match) and for S/B `s2l15` (a *different* design from `v2v_small`/`v2v_base`).
   The notebook loads the official Tiny checkpoint, which is presumably what "imagenet" intends. Training from
   scratch is available with `PRETRAIN="none"`.
5. **Several config keys have no effect.** `PRUNE_THRESHOLD` (0.7 for Market vs 0.25 for others), `FINAL_THRESHOLD`,
   `PRUNE_SIM`, `PRUNE_IMP`, `MERGE_TAU_GATE`, `MERGE_GATE_HIDDEN` and `USE_CHECKPOINT` are never read. The pruning
   threshold comes from the hard-coded `alpha_init=0.7`, and the merge gate from hard-coded `tau_init=0.5`,
   `hidden=64`. So the per-dataset pruning settings in the configs are actually identical.
6. **Config slip:** `configs/OCC_Duke/v2v_tiny.yml` keeps `NAMES: "market1501"`, so as shipped it would train on
   Market. The notebook uses `occ_duke`, as the base/small Occluded-Duke configs do.
7. **Lazy merge gate.** The merge-gate MLP is created on the first forward pass. `train.py` happens to run a debug
   forward pass before building the optimiser, so the MLP exists by then. The notebook reproduces that pass. (Given
   observation 1, it would not train either way.)
8. **The FLOP saving depends on grid packing.** Pruning only shortens the grid width to ⌈K/H⌉, so on a 14×14 grid a
   layer must drop at least 14 tokens (7%) before any compute is saved. The notebook measures the real effect
   (tokens per block and throughput, adaptation on vs off) rather than relying on the abstract's 60% figure.

9. **Minor: checkpoints don't round-trip strictly.** `VSSM._load_from_state_dict` renames any `head.*` key to
   `classifier.head.*` (legacy compatibility), so the model's own unused `base.head` is reported missing on reload.
   The repo's `test.py` loads with `strict=False`. The notebook does the same for resume, but asserts that nothing
   else differs.

## 7. The reproduction notebook

**Faithful to the repo:** model code (embedded verbatim), CUDA kernel source (embedded verbatim), configs (copied),
sampler, augmentations, losses, optimiser parameter groups, LR schedule (exact port, including the repo's
final-epoch quirk where lr drops to 0.1 × lr_min), AMP, seed, evaluation protocol (fp32 features, cosine distance).

**Changes (none affects the maths):**

| Change | Why |
|---|---|
| 3 small patches in the embedded backbone: optional `fvcore`/`mamba2` imports, a chunked PyTorch scan fallback if the CUDA build fails, CPU-safe `torch.cuda.device` context | Kaggle has no `fvcore`; `mamba2` is unused; allows a CPU smoke test |
| Kernel build: explicit `-arch`, C++20 when PyTorch requires it, a force-included header restoring `cub::LaneId`/`cub::CTA_SYNC` on CUDA ≥ 13 | the repo's `setup.py` targets torch 2.2 / CUDA 12 |
| ImageNet VMamba-T weights loaded (see §6.4) | the shipped configs leave the path empty |
| Occluded-Duke and Occluded-REID built in the notebook | not on Kaggle in ready form |
| Resumable state every 5 epochs, clean stop before 12 h, optional mid-training evaluation | Kaggle session limit |
| Efficiency measurement and a "token adaptation off at test" diagnostic | to check the paper's efficiency claim |

**Verified before release:**
* the whole notebook runs end to end on synthetic data in every dataset format (all four datasets, training,
  evaluation, efficiency measurement, results table and figure), including ImageNet-checkpoint loading (0 missing
  backbone tensors);
* the embedded files are byte-identical to the repository's apart from the listed patches (and trailing whitespace
  on blank lines);
* the LR schedule is identical to the repo's at every epoch;
* the CUDA kernel **compiles, links and imports** for the T4 (`sm_75`) using the notebook's own build function
  (tested with CUDA 13.0 + PyTorch 2.14, which needed the C++20 and CUB-compatibility adaptations; it was built
  in about 1 minute). It has not yet been *run* on a GPU here: no GPU was available while preparing this;
* the chunked PyTorch fallback scan matches the repo's reference scan to 5e-6 (outputs and gradients);
* the gradient check behind observations 1–2 (§6);
* stop-and-resume reproduces the uninterrupted run's metrics exactly.

**Expected cost on one Kaggle T4 (estimate; the notebook prints the measured value after the first epoch):**
VMamba-T trains at about 570 img/s on an A100 (VMamba README). A T4 is roughly 5–7× slower, so expect about
80–115 img/s, which gives:

| Dataset | Images × epochs | ≈ T4 hours |
|---|---|---|
| Occluded-REID | 1,000 × 300 | ~1 |
| Market-1501 | 12,936 × 300 | ~9–13 |
| DukeMTMC-reID | 16,522 × 300 | ~12–17 |
| Occluded-Duke | 15,618 × 600 | ~23–33 |

All four together exceed one week of Kaggle's GPU quota (about 30 h). Suggested order: **Occluded-REID** (cheap
sanity check), then **Market-1501** (the paper's headline and efficiency benchmark), then the others. `EPOCH_SCALE`
shortens every schedule if needed; say so when reporting.

### 7.1 Results of the full Kaggle runs (T4, VMamba-Tiny, ImageNet init, full 300-epoch schedule, repository config)

The kernel compiled on Kaggle in 1 min 19 s (torch 2.11 / CUDA 12.8, `-std=c++17`), and the CUDA scan was used.
Training ran at **108 img/s** at steady state. The first epoch is about 2× slower because of warm-up, which made
the first time estimate (23.7 h) far too pessimistic. The notebook now refines it at epoch 6.

| Dataset | Epochs | mAP | R1 | R5 | R10 | Adaptation off at test (mAP / R1) | Tokens / forward | Throughput on vs off | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| Run | Dataset | Epochs | mAP | R1 | R5 | R10 | Adaptation off at test (mAP / R1) | Tokens / forward | Throughput on vs off | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Market-1501 | 300 / 300 | **80.1** | **91.8** | 97.3 | 98.5 | 79.1 / 91.6 | −3.8 % | 250 vs 227 img/s (×1.10) | 9 h 27 min |
| 1 | Occluded-Duke | 38 / 600 | – | – | – | – | – | – | – | stopped at the 11 h budget (not in the paper; abandoned) |
| 2 | Market-1501 | 300 / 300 | **80.5** | **91.7** | 97.1 | 98.6 | 79.2 / 91.4 | −4.2 % | 235 vs 210 img/s (×1.12) | 10 h 05 min |
| 2 | Occluded-REID | 300 / 300 | **84.0** | **88.6** | 96.8 | 98.6 | 83.9 / 89.2 | −4.2 % | 227 vs 204 img/s (×1.11) | 43 min |
| | **Market-1501, mean of runs 1–2** | | **80.3 ± 0.3** | **91.75** | 97.2 | 98.55 | 79.15 | | | |

Run 2 repeats Market-1501 with the same seed (1234). The 0.4 mAP difference is GPU nondeterminism (AMP, atomics in
the scan kernel, cuDNN autotuning), which gives a useful noise floor: differences under about 0.5 mAP are not
meaningful.

mAP over training:
* Market-1501, run 1: 67.8 (ep 50) → 72.1 (100) → 72.4 (150) → 76.6 (200) → 79.1 (250) → 80.1 (300).
* Market-1501, run 2: 68.7 → 74.4 → 75.1 → 75.4 → 79.3 → 80.5.
* Occluded-REID: 82.1 (ep 50) → 82.4 → 81.4 → 82.8 → 83.8 → 84.0. Most of the accuracy is reached by epoch 50,
  since the training set has only 1,000 images.

Training accuracy reaches 1.000 by epoch 250 on both datasets.

**Comparison with the paper:**

| Dataset | Paper (mAP / R1) | Ours (mAP / R1) | Gap |
|---|---|---|---|
| Market-1501, Tiny (Table 2) | 84.2 / 93.6 | 80.3 / 91.75 (mean of 2) | −3.9 / −1.85 |
| Occluded-REID (Table 3, size not stated) | 85.3 / 95.5 | 84.0 / 88.6 | −1.3 / −6.9 |

On Occluded-REID, mAP is close, but R1 is 6.9 points lower. R1 on this dataset depends strongly on the train/test
identity split: only 100 test IDs, with 5 occluded query images and 5 full-body gallery images each. The paper does
not describe its split, and ours is a random half of the IDs (seed 0). The paper also does not say which backbone
size its Occluded-REID number uses; if it is Base, the Tiny gap is expected.

Reading the numbers (paper values from Tables 2, 4 and 5; see §9 for the full comparison):
* **Accuracy:** the paper reports **84.2 mAP / 93.6 R1** for TE-VMamba-Tiny on Market-1501. The mean of our two runs is **−3.9 mAP /
  −1.85 R1**. It follows the repository's config, which differs from the paper's Table 1 in ways that are known to
  cost accuracy: label smoothing off instead of 0.1, batch size 64 instead of 128, and different prune/merge blocks.
* **Efficiency:** the paper reports 5.02 → 1.86 GFLOPs for Tiny (−63 %, Table 4) and ×1.83 throughput for Base
  (Table 5). In our runs, tokens fell 3.8–4.2 % and throughput rose ×1.10–1.12. All of the repository's prune and merge blocks
  (global indices 5–7 and 9–11) sit in stage 3 (14×14 = 196 tokens, dim 384). Pruning keeps about 96 % of tokens
  there and merging about 69 %, while stages 1–2 (3,136 and 784 tokens) are untouched. Weighted by per-token cost
  (∝ dim²), stage 3 is roughly 55–60 % of the backbone FLOPs, so this configuration cannot reach a 63 % saving.
  The paper's own Table 4 attributes the whole saving to pruning (merging alone leaves FLOPs at 5.02). With the
  repository's fixed threshold θ = μ − 1.10σ (α = softplus(0.7), never trained; observation 1), only about 4 % of
  tokens are pruned.
* **The adaptation costs no accuracy in our run:** with the same weights, turning it off at test gives 79.1 vs
  80.1 mAP. In the paper's Table 4, by contrast, the variant without token reduction is much better (Tiny: 91.6
  mAP / 98.2 R1 vs 84.2 / 93.6).

## 8. Suggested follow-up experiments
* **Fix observation 1–2 and measure the difference:** remove the `torch.no_grad()` around prune/merge, so the
  thresholds and the frozen SS2D layers train. This tests whether the paper's "learnable thresholds" matter.
* **Token adaptation off at train and test** (`TOKEN_ADAPT.ENABLE=False`): the plain VMamba baseline the paper
  compares against.
* **From scratch vs ImageNet** (`PRETRAIN="none"`): what the shipped configs actually do.

## 9. Paper vs repository vs this reproduction

### 9.1 What the paper actually evaluates
Market-1501, CUHK03-NP (labeled and detected), MSMT17 and Occluded-REID, on one RTX 5080 (§5.1). **DukeMTMC-reID and
Occluded-Duke are not in the paper**: they appear only in the repository's configs. So runs on those two datasets
have nothing in the paper to compare with.

| Paper result (no re-ranking) | Tiny | Small | Base |
|---|---|---|---|
| Market-1501 R1 / mAP (Table 2) | 93.6 / 84.2 | 95.1 / 85.7 | 96.7 / 87.1 |
| CUHK03-NP labeled R1 / mAP | 96.3 / 92.1 | 95.5 / 91.7 | 97.3 / 94.0 |
| CUHK03-NP detected R1 / mAP | 94.0 / 90.6 | 93.2 / 89.5 | 95.4 / 91.7 |
| MSMT17 R1 / mAP | 82.8 / 62.9 | 80.9 / 60.9 | 83.3 / 62.7 |
| Occluded-REID R1 / mAP (Table 3) | 95.5 / 85.3, size not stated | | |
| Market GFLOPs, none → prune+merge (Table 4) | 5.02 → 1.86 | 8.79 → 2.20 | 15.46 → 3.82 |
| Market, no token reduction R1 / mAP (Table 4) | 98.2 / 91.6 | 98.1 / 91.1 | 98.2 / 91.6 |

### 9.2 Settings: paper (Table 1) vs repository (`v2v_tiny.yml`) vs our run

| Setting | Paper, Table 1 | Repository config | Our run |
|---|---|---|---|
| Input / patch | 224 / 4 | 224 / 4 | 224 / 4 |
| Tiny dims / depths | 96 / [2,2,8,2] | 96 / [2,2,8,2] | same |
| Pruning layers | 5–10 | 5, 6, 7 | 5, 6, 7 |
| Pruning α | 0.7 | softplus(0.7) = 1.10, hard-coded, not trained | same as repository |
| Merging layers | 12–18 (beyond Tiny's 14 blocks, so presumably given for Base) | 9, 10, 11 | 9, 10, 11 |
| Merging threshold τ | 0.1 | sigmoid(MLP), with τ_init 0.5 hard-coded | same as repository |
| Batch size | **128** | **64** (16 × 4) | 64 |
| Label smoothing | **0.1** | **off** | off |
| Optimiser | AdamW, lr 3e-4, wd 0.05 | same | same |
| Loss | CE + triplet (λ = 1, margin 0.3) | same, but no smoothing | same as repository |
| Epochs | **not stated** | 300 (600 for Occluded-Duke and MSMT17) | 300 |
| Initialisation | not stated | "imagenet" with an empty path, so nothing is loaded | official VMamba-T ImageNet-1K |

### 9.3 Points worth raising
1. **The accuracy–efficiency trade-off is larger than the abstract suggests.** In Table 4, TE-VMamba-Tiny loses
   7.4 mAP and 4.6 R1 against the same backbone without token reduction (91.6 → 84.2 mAP). The "competitive
   accuracy" in the abstract is measured against other methods, not against the unreduced backbone.
2. **The no-reduction baselines are unusually strong.** 91.6 mAP / 98.2 R1 on Market-1501 without re-ranking is
   above every method in Table 2, including TransReID (88.9) and DC-Former (90.6). The baseline is also almost
   identical for Tiny, Small and Base (91.6 / 91.1 / 91.6). Asking the authors how these numbers were obtained is
   reasonable.
3. **The CUHK03-NP numbers are high.** 92.1 mAP (labeled, Tiny) is well above the Transformer methods in Table 2
   (79.6–83.3 mAP).
4. **The released code does not match the stated method.** The thresholds are fixed (§6, observation 1), α
   differs (1.10 vs 0.7), and the prune and merge layers differ from Table 1. The ~4 % pruning rate we measured
   explains why the 63 % FLOPs saving is not reproduced.
5. **Comparable datasets in this notebook:** Market-1501 and Occluded-REID. MSMT17 images are not on Kaggle. CUHK03
   is on Kaggle, but the repository has no CUHK03 loader, so one would have to be written.

### 9.4 `PROTOCOL = "paper"`: running with the paper's Table 1 settings
The notebook has a switch (`PROTOCOL`, run-configuration cell) that applies the paper's Table 1 wherever it
differs from the repository config. Everything else stays as in the repository, because the paper does not state
it: 300 epochs, 4 images per identity, warm-up and cosine schedule, drop-path 0.2, augmentations, ImageNet
initialisation.

| Setting | `PROTOCOL="repo"` | `PROTOCOL="paper"` |
|---|---|---|
| Label smoothing | off | **0.1** |
| Batch | 64 (16 IDs × 4) | **128 (32 IDs × 4)** |
| Pruning blocks | 5–7 | **5–10** |
| α | softplus(0.7) = 1.10 | **0.7** (raw parameter = softplus⁻¹(0.7)) |
| Merging blocks | 9–11 | **12–13**: Table 1 says 12–18, but VMamba-Tiny has only blocks 0–13 |
| Merging threshold τ | 0.5 | **0.1** (sigmoid(θ_base) = 0.1) |

How it is implemented:
* α and τ are hard-coded in the authors' `VSSBlock`, so they are set on the built modules after the ImageNet
  weights load. The embedded source is not edited.
* Results go to `<dataset>_paper/`, so they never mix with the repo-protocol runs.
* Batch 128 is first tried on one T4 (the log prints peak memory). If it does not fit and a second GPU is visible,
  training switches to `nn.DataParallel`. The loss is still computed on GPU 0 over all 128 images, so batch-hard
  triplet mining is unchanged.

Checked locally (CPU, 224×224, ImageNet weights):
* α = 0.700 in all six prune modules and τ = 0.100 in both merge modules.
* An end-to-end run, and a stop-and-resume run, give identical metrics.
* What we could not check without a GPU: the memory probe and the DataParallel fallback.

**What to expect on efficiency.** The paper settings do not make the token reduction large:
* With α = 0.7, pruning still keeps about 95 % of tokens. |Δ| has a long tail, so σ is large and μ − 0.7σ sits near
  the bottom of the distribution.
* On a 14-row grid, a layer saves compute only when it drops at least 14 tokens.
* τ = 0.1 merges almost every pair, but only in the last stage (7×7 → 21 tokens).
* On the same images, the FLOPs proxy falls by about 6 % (repo protocol: about 13 %), against the paper's 63 %.

So this run mainly tests whether **label smoothing and batch 128** close the accuracy gap (84.2 vs 80.3 mAP). The
efficiency claim would need a different pruning rule from the released one (for example one that actually trains α,
which the hard keep-mask and `no_grad` prevent).
