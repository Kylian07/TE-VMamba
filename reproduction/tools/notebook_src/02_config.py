# %% [markdown]
# ## Configuration: port of `config/defaults.py` + `configs/<dataset>/v2v_<size>.yml`
# Values below are copied from the repo's YAML files. Comments mark the keys that the code **never reads**: they are
# kept for completeness but have no effect, exactly as in the original.

# %%
class CN(dict):
    """Tiny attribute-dict so ported code can keep `cfg.SOLVER.BASE_LR` syntax."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(d):
        return CN({k: CN.wrap(v) if isinstance(v, dict) else v for k, v in d.items()})


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if (k in out and isinstance(out[k], dict) and isinstance(v, dict)) else copy.deepcopy(v)
    return out


BASE = {   # == configs/Market/v2v_base.yml merged over config/defaults.py
    "MODEL": {
        "NAME": "vssm", "PRETRAIN_CHOICE": "imagenet", "PRETRAIN_PATH": "", "DROP_PATH_RATE": 0.5,
        "NECK": "bnneck", "COS_LAYER": False, "ID_LOSS_TYPE": "softmax", "METRIC_LOSS_TYPE": "triplet",
        "IF_LABELSMOOTH": "off", "IF_WITH_CENTER": "no", "NO_MARGIN": False,
        "ID_LOSS_WEIGHT": 1.0, "TRIPLET_LOSS_WEIGHT": 1.0,
        "VSSM": {"IN_CHANS": 3, "PATCH_SIZE": 4, "EMBED_DIM": 128, "DEPTHS": [2, 2, 20, 2], "SSM_RATIO": 1.0,
                 "SSM_D_STATE": 1, "SSM_DT_RANK": "auto", "SSM_CONV": 3, "SSM_CONV_BIAS": False,
                 "SSM_FORWARDTYPE": "v05_noz", "MLP_RATIO": 4.0, "DOWNSAMPLE": "v3", "PATCHEMBED": "v2",
                 "NORM_LAYER": "ln2d"},
        "TOKEN_ADAPT": {
            "ENABLE": True, "TOKEN_PRUNE": True, "TOKEN_MERGE": True,
            "PRUNE_LAYERS": [5, 6, 7], "MERGE_LAYERS": [12, 13, 14, 15, 16], "MERGE_START": 8,
            "PRUNE_THRESHOLD": 0.7,     # NOT READ: LayerTokenPrune is built with alpha_init=0.7 hard-coded
            "PRUNE_SIM": "cos_delta",   # NOT READ
            "PRUNE_IMP": "l2",          # NOT READ
            "MERGE_TAU_GATE": 0.5,      # NOT READ (tau_init=0.5 hard-coded, same value)
            "MERGE_GATE_HIDDEN": 64,    # NOT READ (64 hard-coded, same value)
            "FINAL_THRESHOLD": 0.1,     # NOT READ by any module
        },
    },
    "INPUT": {"SIZE_TRAIN": [224, 224], "SIZE_TEST": [224, 224], "PROB": 0.5, "RE_PROB": 0.3, "PADDING": 10,
              "PIXEL_MEAN": [0.485, 0.456, 0.406], "PIXEL_STD": [0.229, 0.224, 0.225]},
    "AUG": {"COLOR_JITTER": 0.0, "AUTO_AUGMENT": "none"},
    "DATASETS": {"NAMES": "market1501", "ROOT_DIR": DATA_ROOT},
    "DATALOADER": {"SAMPLER": "softmax_triplet", "NUM_INSTANCE": 4, "NUM_WORKERS": NUM_WORKERS},
    "SOLVER": {"OPTIMIZER_NAME": "AdamW", "MAX_EPOCHS": 300, "BASE_LR": 3e-4, "IMS_PER_BATCH": 64,
               "WARMUP_EPOCHS": 5, "WARMUP_METHOD": "linear", "LARGE_FC_LR": False, "BIAS_LR_FACTOR": 1,
               "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.05, "WEIGHT_DECAY_BIAS": 0.0, "SEED": 1234, "MARGIN": 0.3,
               "CENTER_LR": 0.5, "CENTER_LOSS_WEIGHT": 0.0005, "CHECKPOINT_PERIOD": 50, "LOG_PERIOD": 100,
               "EVAL_PERIOD": 300},
    "TEST": {"IMS_PER_BATCH": 256, "RE_RANKING": False, "NECK_FEAT": "before", "FEAT_NORM": "yes"},
}

SIZE_OVERRIDES = {   # differences of configs/*/v2v_{tiny,small}.yml from v2v_base.yml
    "base": {},
    "small": {"MODEL": {"DROP_PATH_RATE": 0.3, "VSSM": {"EMBED_DIM": 96}}},
    "tiny": {"MODEL": {"DROP_PATH_RATE": 0.2, "VSSM": {"EMBED_DIM": 96, "DEPTHS": [2, 2, 8, 2]},
                       "TOKEN_ADAPT": {"MERGE_LAYERS": [9, 10, 11]}}},
}

DATASET_OVERRIDES = {   # differences of configs/<dataset>/v2v_*.yml from the Market file
    "market1501": {},
    "dukemtmc": {"DATASETS": {"NAMES": "dukemtmc"}, "MODEL": {"TOKEN_ADAPT": {"PRUNE_THRESHOLD": 0.25}}},
    # NOTE: the shipped configs/OCC_Duke/v2v_tiny.yml keeps NAMES: "market1501" (a copy-paste slip); the base/small
    # Occluded-Duke configs use "occ_duke", which is clearly the intent, so it is used here.
    "occ_duke": {"DATASETS": {"NAMES": "occ_duke"}, "MODEL": {"TOKEN_ADAPT": {"PRUNE_THRESHOLD": 0.25}},
                 "SOLVER": {"MAX_EPOCHS": 600, "EVAL_PERIOD": 600}},
    "occ_reid": {"DATASETS": {"NAMES": "occ_reid"}, "MODEL": {"TOKEN_ADAPT": {"PRUNE_THRESHOLD": 0.25}}},
}

# PROTOCOL = "paper": the paper's Table 1 ("all parameters are equally applied" to Tiny/Small/Base), where it
# differs from the repository's configs. Not stated in the paper, so kept from the repo: epochs, NUM_INSTANCE (4 images
# per ID, so batch 128 = 32 IDs x 4), warm-up/cosine schedule, drop-path, augmentations.
#   * label smoothing 0.1 (repo: off)          * batch 128 (repo: 64)
#   * pruning blocks 5-10 (repo: 5-7)          * alpha = 0.7, i.e. raw parameter softplus^-1(0.7) (repo: raw 0.7 -> 1.10)
#   * merging blocks 12-18 (repo Tiny: 9-11). VMamba-Tiny has blocks 0-13, so only 12-13 (stage 4) exist.
#   * merging threshold tau = 0.1, i.e. sigmoid(theta_base) = 0.1 (repo: tau_init = 0.5 hard-coded in VSSBlock)
# alpha and tau are hard-coded inside the authors' VSSBlock, so they are set on the built modules (07_model), without
# editing the embedded source.
PAPER_TABLE1 = {"MODEL": {"IF_LABELSMOOTH": "on",
                          "TOKEN_ADAPT": {"PRUNE_LAYERS": list(range(5, 11)), "MERGE_LAYERS": list(range(12, 19)),
                                          "PRUNE_ALPHA": 0.7, "MERGE_TAU": 0.1}},
                "SOLVER": {"IMS_PER_BATCH": 128}}


def run_key(dataset):
    """Output folder name: '<dataset>' for the repo protocol, '<dataset>_paper' for the paper protocol."""
    return dataset if PROTOCOL == "repo" else f"{dataset}_{PROTOCOL}"


# Official VMamba ImageNet-1K checkpoints (MzeroMiko/VMamba releases). Only VMamba-T [s1l8] has exactly the same
# architecture as the repo's v2v configs; the released S/B checkpoints are the s2l15 design (depths 2-2-15-2,
# ssm_ratio 2.0), which does not match v2v_small/v2v_base, so those sizes fall back to random init.
IMAGENET_CKPTS = {
    "tiny": ("vssm1_tiny_0230s_ckpt_epoch_264.pth",
             "https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth"),
}


def build_cfg(dataset):
    raw = _merge(_merge(BASE, SIZE_OVERRIDES[MODEL_SIZE]), DATASET_OVERRIDES[dataset])
    if PROTOCOL == "paper":
        raw = _merge(raw, PAPER_TABLE1)
        n_blocks = sum(raw["MODEL"]["VSSM"]["DEPTHS"])
        ta = raw["MODEL"]["TOKEN_ADAPT"]
        ta["PRUNE_LAYERS"] = [i for i in ta["PRUNE_LAYERS"] if i < n_blocks]
        ta["MERGE_LAYERS"] = [i for i in ta["MERGE_LAYERS"] if i < n_blocks]   # Tiny: [12, 13]
    elif PROTOCOL != "repo":
        raise ValueError(f"PROTOCOL must be 'repo' or 'paper', got {PROTOCOL!r}")
    cfg = CN.wrap(raw)
    cfg.SOLVER.MAX_EPOCHS = max(1, int(round(cfg.SOLVER.MAX_EPOCHS * EPOCH_SCALE)))
    if RUN_MODE == "quick":
        cfg.SOLVER.MAX_EPOCHS = QUICK["epochs"]
        cfg.SOLVER.WARMUP_EPOCHS = 1
        cfg.SOLVER.LOG_PERIOD = 5
    if TEST_IMG_SIZE:
        s = int(TEST_IMG_SIZE)
        cfg.INPUT.SIZE_TRAIN = cfg.INPUT.SIZE_TEST = [s, s]
        cfg.SOLVER.IMS_PER_BATCH = 16 if PROTOCOL == "repo" else 32
    return cfg


_c = build_cfg("market1501")
print(f"PROTOCOL={PROTOCOL}: batch {_c.SOLVER.IMS_PER_BATCH} | label smoothing {_c.MODEL.IF_LABELSMOOTH} | "
      f"prune blocks {_c.MODEL.TOKEN_ADAPT.PRUNE_LAYERS} | merge blocks {_c.MODEL.TOKEN_ADAPT.MERGE_LAYERS} | "
      f"alpha {_c.MODEL.TOKEN_ADAPT.get('PRUNE_ALPHA') or 'softplus(0.7)=1.10 (hard-coded)'} | "
      f"tau {_c.MODEL.TOKEN_ADAPT.get('MERGE_TAU') or '0.5 (hard-coded)'} | epochs {_c.SOLVER.MAX_EPOCHS}")
