# %% [markdown]
# ## ReID model (port of `model/make_model.py`)
# The VSSM backbone feeds global average pooling, then a BNNeck (BatchNorm1d), then a linear ID classifier.
# * Training returns (ID logits, pooled feature). Triplet loss uses the feature *before* the BNNeck.
# * Testing returns the pooled feature *before* the BNNeck (`TEST.NECK_FEAT = before`), L2-normalised, cosine distance.
#
# **ImageNet initialisation.** The repo's configs say `PRETRAIN_CHOICE: imagenet` but leave `PRETRAIN_PATH` empty, so
# the released configs actually train from random weights. Here the official **VMamba-T [s1l8] ImageNet-1K
# checkpoint** (82.6% top-1) is loaded when `MODEL_SIZE="tiny"`. It has exactly the v2v_tiny architecture. The download
# comes from the VMamba authors' release page; a copy attached as a Kaggle input is used first if present.

# %%
def weights_init_kaiming(m):
    cn = m.__class__.__name__
    if cn.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out"); nn.init.constant_(m.bias, 0.0)
    elif cn.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif cn.find("BatchNorm") != -1 and getattr(m, "affine", False):
        nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if m.__class__.__name__.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def resolve_pretrain_path():
    if PRETRAIN != "imagenet":
        return ""
    if MODEL_SIZE not in IMAGENET_CKPTS:
        print(f"[pretrain] no official ImageNet checkpoint matches MODEL_SIZE={MODEL_SIZE} (v2v design) -> random init")
        return ""
    fname, url = IMAGENET_CKPTS[MODEL_SIZE]
    for d in ALL_INPUT_DIRS:
        if osp.isfile(osp.join(d, fname)):
            return osp.join(d, fname)
    try:
        return _download(url, osp.join(CACHE_DIR, fname))
    except Exception as e:  # noqa: BLE001
        print(f"!! [pretrain] download failed ({type(e).__name__}: {e}). Attach {fname} as a Kaggle input. -> random init")
        return ""


class VMambaReID(nn.Module):
    def __init__(self, num_classes, cfg, pretrain_path=""):
        super().__init__()
        self.neck, self.neck_feat, self.ID_LOSS_TYPE = cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.ID_LOSS_TYPE
        vcfg, ta = cfg.MODEL.VSSM, cfg.MODEL.TOKEN_ADAPT
        vmamba_mod.ENABLE_TOKEN_ADAPTATION = bool(ta.ENABLE)
        self.base = VSSM(
            patch_size=vcfg.PATCH_SIZE, in_chans=vcfg.IN_CHANS, num_classes=0, depths=vcfg.DEPTHS,
            dims=[vcfg.EMBED_DIM, vcfg.EMBED_DIM * 2, vcfg.EMBED_DIM * 4, vcfg.EMBED_DIM * 8],
            ssm_d_state=vcfg.SSM_D_STATE, ssm_dt_rank=vcfg.SSM_DT_RANK, ssm_ratio=vcfg.SSM_RATIO,
            ssm_conv=vcfg.SSM_CONV, ssm_conv_bias=vcfg.SSM_CONV_BIAS, forward_type=vcfg.SSM_FORWARDTYPE,
            mlp_ratio=vcfg.MLP_RATIO, drop_path_rate=cfg.MODEL.DROP_PATH_RATE, norm_layer=vcfg.NORM_LAYER,
            downsample_version=vcfg.DOWNSAMPLE, patchembed_version=vcfg.PATCHEMBED, imgsize=cfg.INPUT.SIZE_TRAIN[0],
            token_merge=bool(ta.ENABLE and ta.TOKEN_MERGE), token_prune=bool(ta.ENABLE and ta.TOKEN_PRUNE),
            merge_layers=list(ta.MERGE_LAYERS), prune_layers=list(ta.PRUNE_LAYERS), merge_start=int(ta.MERGE_START),
            prune_threshold=float(ta.PRUNE_THRESHOLD), prune_sim=str(ta.PRUNE_SIM), prune_imp=str(ta.PRUNE_IMP),
            final_threshold=float(ta.FINAL_THRESHOLD),
        )
        self.channel_first = self.base.channel_first
        self.in_planes, self.num_classes = self.base.num_features, num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.pretrain_report = "random init"
        if cfg.MODEL.PRETRAIN_CHOICE in ("imagenet", "pretrained") and pretrain_path:
            self._load_backbone_pretrained(pretrain_path)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward_features(self, x):
        x = self.base.patch_embed(x)
        if self.base.pos_embed is not None:
            pos = self.base.pos_embed
            x = x + (pos.permute(0, 2, 3, 1) if not self.base.channel_first else pos)
        for layer in self.base.layers:
            x = layer(x)
        return x

    def pool(self, feat_map):
        if not self.channel_first:
            feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        return F.adaptive_avg_pool2d(feat_map, 1).flatten(1)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.pool(self.forward_features(x))
        feat = self.bottleneck(global_feat) if self.neck == "bnneck" else global_feat
        if self.training:
            return self.classifier(feat), global_feat
        return feat if self.neck_feat == "after" else global_feat

    def _load_backbone_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            ckpt = ckpt.get("state_dict", ckpt.get("model", ckpt))
        new = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        for k in list(new):
            if k.startswith("classifier.head.") or k.startswith("head."):
                new.pop(k)
        inc = self.base.load_state_dict(new, strict=False)
        real_missing = [k for k in inc.missing_keys if "prune_module" not in k and "merge_module" not in k
                        and not k.startswith(("head.", "classifier."))]
        self.pretrain_report = (f"loaded {osp.basename(path)}: {len(new) - len(inc.unexpected_keys)} tensors, "
                                f"missing backbone tensors={len(real_missing)}, unexpected={len(inc.unexpected_keys)}")
        print("[pretrain]", self.pretrain_report)
        if real_missing:
            print("   missing (first 10):", real_missing[:10])


def token_stats(model):
    """Average keep ratios reported by the prune / merge modules after the last forward pass."""
    pr, mr = [], []
    for m in model.modules():
        if m.__class__.__name__ == "LayerTokenPrune" and m.keep_ratio_est is not None:
            pr.append(m.keep_ratio_est)
        if m.__class__.__name__ == "FixedPairThresholdMerge" and m.keep_ratio_est is not None:
            mr.append(m.keep_ratio_est)
    return pr, mr


# %% [markdown]
# ## Losses, optimiser, LR schedule (port of `loss/`, `solver/make_optimizer.py`, `solver/scheduler_factory.py`)
# * Loss = ID cross-entropy (label smoothing **off** in the configs) + batch-hard triplet (margin 0.3) on the
#   pre-BNNeck feature, each weighted 1.0.
# * AdamW, lr 3e-4. Weight decay 0.05 on weights, 0 on biases, norms, and any parameter whose name contains "norm",
#   "bn" or "ln" (the repo's name heuristic).
# * Cosine schedule over all epochs, linear warm-up for 5 epochs from 1% of lr, floor at 0.2% of lr, stepped once per
#   epoch (exact port of the repo's modified timm `CosineLRScheduler`).

# %%
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    return (xx + yy - 2 * torch.matmul(x, y.t())).clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels):
    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss:
    def __init__(self, margin=None):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) if margin is not None else nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        dist_ap, dist_an = hard_example_mining(euclidean_dist(global_feat, global_feat), labels)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y) if self.margin is not None else self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes, self.epsilon, self.logsoftmax = num_classes, epsilon, nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        t = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        t = (1 - self.epsilon) * t + self.epsilon / self.num_classes
        return (-t * log_probs).mean(0).sum()


def make_loss(cfg, num_classes):
    triplet = TripletLoss() if cfg.MODEL.NO_MARGIN else TripletLoss(cfg.SOLVER.MARGIN)
    xent = CrossEntropyLabelSmooth(num_classes) if cfg.MODEL.IF_LABELSMOOTH == "on" else None

    def loss_func(score, feat, target, target_cam=None):
        id_loss = xent(score, target) if xent is not None else F.cross_entropy(score, target)
        tri_loss = triplet(feat, target)[0]
        return cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss
    return loss_func


def make_optimizer(cfg, model):
    base_lr, wd = cfg.SOLVER.BASE_LR, cfg.SOLVER.WEIGHT_DECAY
    nwd_names, nwd_keywords = set(), set()
    for obj in (model, getattr(model, "base", None)):
        if obj is not None and hasattr(obj, "no_weight_decay"):
            nwd_names |= set(obj.no_weight_decay())
        if obj is not None and hasattr(obj, "no_weight_decay_keywords"):
            nwd_keywords |= set(obj.no_weight_decay_keywords())

    def no_wd(name):
        return (name in nwd_names or any(kw and kw in name for kw in nwd_keywords) or name.endswith(".bias")
                or any(x in name.lower() for x in ["norm", "bn", "ln", "layernorm", "batchnorm"]))

    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (nodecay if no_wd(name) else decay).append(p)
    groups = [{"params": decay, "lr": base_lr, "weight_decay": wd}, {"params": nodecay, "lr": base_lr, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=0.0)


class RepoCosineLR:
    """Exact port of solver/cosine_lr.py + solver/scheduler_factory.py as configured by the repo
    (t_mul=1, decay_rate=0.1, warmup_prefix=False, cycle_limit=1, no noise). Note the repo quirk kept here: at the
    final epoch (t == t_initial) the cycle has ended and lr drops to lr_min * decay_rate (= 6e-8), whereas timm's
    built-in scheduler would keep lr_min."""
    def __init__(self, optimizer, t_initial, warmup_t, lr_min_ratio=0.002, warmup_ratio=0.01, decay_rate=0.1):
        self.optimizer, self.t_initial, self.warmup_t, self.decay_rate = optimizer, t_initial, int(warmup_t), decay_rate
        self.base_values = [g["lr"] for g in optimizer.param_groups]
        self.lr_min = [lr_min_ratio * b for b in self.base_values]
        self.warmup_lr_init = [warmup_ratio * b for b in self.base_values]
        self.warmup_steps = ([(b - w) / self.warmup_t for b, w in zip(self.base_values, self.warmup_lr_init)]
                             if self.warmup_t else [1.0] * len(self.base_values))
        if self.warmup_t:
            self._set(self.warmup_lr_init)

    def _set(self, lrs):
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g["lr"] = lr

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [w + t * s for w, s in zip(self.warmup_lr_init, self.warmup_steps)]
        i = t // self.t_initial
        t_curr = t - self.t_initial * i
        gamma = self.decay_rate ** i
        lr_max = [v * gamma for v in self.base_values]
        lr_min = [v * gamma for v in self.lr_min]
        if i < 1:   # cycle_limit = 1
            return [lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * t_curr / self.t_initial)) for hi, lo in zip(lr_max, lr_min)]
        return lr_min

    def step(self, epoch):
        self._set(self._get_lr(epoch))

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


def create_scheduler(cfg, optimizer):
    return RepoCosineLR(optimizer, t_initial=cfg.SOLVER.MAX_EPOCHS, warmup_t=cfg.SOLVER.WARMUP_EPOCHS)
