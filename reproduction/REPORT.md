# TE-VMamba: SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging for Person Re-ID
### A plain-language methodology report and reproduction guide

> **Paper:** Huiyuan Huang, Sang Min Yoon. *SSM-Aware Token-Efficient VMamba via Adaptive Patch Pruning and Merging
> for Person Re-Identification.* CVPR 2026 (poster). Code: this repository (`TEVmamba/`).
>
> **Companion notebook:** [`TE-VMamba_Kaggle_Reproduction.ipynb`](TE-VMamba_Kaggle_Reproduction.ipynb) is a single
> Kaggle notebook that embeds the authors' backbone and CUDA kernel, prepares the datasets from Kaggle and official
> sources, and trains and evaluates with the repository's protocol.
>
> **How this report was made:** from a line-by-line reading of the code in this repository, plus the paper's public
> abstract. The paper PDF could not be opened from the environment used to prepare this report (CVF Open Access was
> blocked), so **no paper result numbers are quoted here**. Copy them from the paper's tables into
> `PAPER_REPORTED` in the notebook. Everything stated about *what the code does* was checked against the code, and
> the key claims were also checked by running it (§6).

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

### 7.1 Results of the first full Kaggle run (T4, VMamba-Tiny, ImageNet init, full 300-epoch schedule)

The kernel compiled on Kaggle in 1 min 19 s (torch 2.11 / CUDA 12.8, `-std=c++17`), and the CUDA scan was used.
Training ran at **108 img/s** at steady state. The first epoch is about 2× slower because of warm-up, which made
the first time estimate (23.7 h) far too pessimistic. The notebook now refines it at epoch 6.

| Dataset | Epochs | mAP | R1 | R5 | R10 | Adaptation off at test (mAP / R1) | Tokens / forward | Throughput on vs off | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| Market-1501 | 300 / 300 | **80.1** | **91.8** | 97.3 | 98.5 | 79.1 / 91.6 | −3.8 % | 250 vs 227 img/s (×1.10) | 9 h 27 min |
| Occluded-Duke | 38 / 600 | – | – | – | – | – | – | – | stopped at the 11 h budget; ~20 h left (≈2 sessions) |

Market-1501 mAP over training: 67.8 (ep 50) → 72.1 (100) → 72.4 (150) → 76.6 (200) → 79.1 (250) → 80.1 (300).
Training accuracy reaches 1.000 and loss 0.001 by epoch 250.

Reading the numbers:
* **Accuracy:** this is the repo's Tiny protocol, a single run with seed 1234. Compare it with the paper's
  TE-VMamba-Tiny Market-1501 row (fill `PAPER_REPORTED`). Strong baselines in the literature (BoT ResNet-50
  ≈ 86 mAP, TransReID ViT-B ≈ 89 mAP) use 256×128 input, label smoothing and/or larger backbones, so they are not
  like-for-like with a 224×224 VMamba-T run without label smoothing.
* **Efficiency:** the reduction is small. All prune and merge blocks (global indices 5–7 and 9–11) sit in stage 3
  (14×14 = 196 tokens, dim 384). Pruning keeps about 96 % and merging about 69 % of the tokens there, while stages
  1–2 (3,136 and 784 tokens) are untouched, so the token total falls only 3.8 %. Weighting by per-token cost
  (∝ dim²), stage 3 is roughly 55–60 % of the backbone FLOPs. That bounds the possible saving in this configuration
  at about 10–20 %. A secondary summary of the paper mentions "over 60 % FLOPs reduction" on Market-1501. If the
  paper says that for this Tiny setting, this run does not reproduce it. A likely contributor is observation 1–2
  (§6): the `torch.no_grad()` around prune/merge means the pruning thresholds never train and stay at their
  initial ~4 % pruning rate.
* **The adaptation costs no accuracy:** with the same weights, turning it off at test gives 79.1 vs 80.1 mAP.

## 8. Suggested follow-up experiments
* **Fix observation 1–2 and measure the difference:** remove the `torch.no_grad()` around prune/merge, so the
  thresholds and the frozen SS2D layers train. This tests whether the paper's "learnable thresholds" matter.
* **Token adaptation off at train and test** (`TOKEN_ADAPT.ENABLE=False`): the plain VMamba baseline the paper
  compares against.
* **From scratch vs ImageNet** (`PRETRAIN="none"`): what the shipped configs actually do.
