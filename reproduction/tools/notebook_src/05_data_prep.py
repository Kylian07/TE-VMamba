# %% [markdown]
# ## Data preparation
# The notebook finds the attached Kaggle datasets by their folder signature and builds the layout the repo's loaders
# expect under `DATA_ROOT`:
# ```
# DATA_ROOT/market1501/{bounding_box_train, query, bounding_box_test}       <- symlinks to the Kaggle input
# DATA_ROOT/dukemtmcreid/{...}                                               <- symlinks
# DATA_ROOT/Occluded_Duke/{...}                                              <- built from DukeMTMC + official lists
# DATA_ROOT/Occluded_REID/{train/<pid>, test/query/<pid>, test/gallery/<pid>}/*.jpg   <- converted from the zip
# ```

# %%
def _list_dirs(root, max_depth=11):
    out, base = [], osp.abspath(root).rstrip("/").count("/")
    for cur, dirs, _ in os.walk(root, followlinks=True):
        if cur.rstrip("/").count("/") - base >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        out.append(cur)
    return out


ALL_INPUT_DIRS = _list_dirs(INPUT_ROOT) if osp.isdir(INPUT_ROOT) else []
print(f"scanned {len(ALL_INPUT_DIRS)} directories under {INPUT_ROOT}")


def _has_images(d, exts=(".jpg", ".jpeg", ".png")):
    try:
        return any(e.name.lower().endswith(exts) for e in os.scandir(d))
    except OSError:
        return False


def _find(pred, keyword=None):
    c = [d for d in ALL_INPUT_DIRS if (keyword is None or keyword in d.lower()) and pred(d)]
    return sorted(c, key=lambda p: (p.count("/"), len(p)))[0] if c else None


def _market_style(d):
    return all(_has_images(osp.join(d, s)) for s in ("bounding_box_train", "query", "bounding_box_test"))


def _symlink(src, dst):
    if osp.islink(dst):
        os.unlink(dst)
    elif osp.exists(dst):
        return
    os.makedirs(osp.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def _download(url, out, sha256=None, timeout=300):
    if not osp.isfile(out):
        print(f"   downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TE-VMamba reproduction notebook)"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(out + ".part", "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        os.replace(out + ".part", out)
    if sha256 and hashlib.sha256(open(out, "rb").read()).hexdigest() != sha256:
        os.remove(out)
        raise RuntimeError(f"sha256 mismatch for {out}")
    return out


PREPARED, PREP_NOTES = {}, {}

# ---- Market-1501 / DukeMTMC-reID ---------------------------------------------------------------------------------------
for name, kw in (("market1501", "market"), ("dukemtmcreid", "duke")):
    src = DATA_OVERRIDES.get(name) or _find(lambda d: _market_style(d) and "occluded" not in d.lower(), kw)
    if src:
        for s in ("bounding_box_train", "query", "bounding_box_test"):
            _symlink(osp.join(src, s), osp.join(DATA_ROOT, name, s))
        PREPARED[name] = osp.join(DATA_ROOT, name)
        print(f"[data] {name:14s} <- {src}")
if "dukemtmcreid" in PREPARED:
    PREPARED["dukemtmc"] = PREPARED["dukemtmcreid"]

# ---- Occluded-Duke: official name lists applied to DukeMTMC-reID (== lightas/convert_duke_to_occduke.py) --------------
OCCDUKE_LIST_URL = "https://raw.githubusercontent.com/lightas/Occluded-DukeMTMC-Dataset/master/Occluded_Duke/{}.list"


def build_occluded_duke(duke_dir, out_dir):
    if _market_style(out_dir) and len(os.listdir(osp.join(out_dir, "query"))) == 2210:
        return "already built"
    lists = {}
    for split in ("train", "query", "gallery"):
        attached = _find(lambda d: osp.isfile(osp.join(d, f"{split}.list")))
        p = osp.join(attached, f"{split}.list") if attached else _download(
            OCCDUKE_LIST_URL.format(split), osp.join(CACHE_DIR, f"occduke_{split}.list"))
        lists[split] = [l.strip() for l in open(p) if l.strip()]
    missing = 0
    for split, folder in (("train", "bounding_box_train"), ("gallery", "bounding_box_test"), ("query", "query")):
        os.makedirs(osp.join(out_dir, folder), exist_ok=True)
        for img in lists[split]:
            src = osp.join(duke_dir, folder, img)
            if not osp.isfile(src):                       # image moved to another split by the re-split
                src = osp.join(duke_dir, "bounding_box_test", img)
            if not osp.isfile(src):
                src = osp.join(duke_dir, "query", img)
            if osp.isfile(src):
                _symlink(src, osp.join(out_dir, folder, img))
            else:
                missing += 1
    return f"built (train {len(lists['train'])}, query {len(lists['query'])}, gallery {len(lists['gallery'])}, missing {missing})"


if "dukemtmcreid" in PREPARED:
    try:
        msg = build_occluded_duke(osp.realpath(osp.join(PREPARED["dukemtmcreid"], "bounding_box_train", "..")),
                                  osp.join(DATA_ROOT, "Occluded_Duke"))
        PREPARED["occ_duke"] = osp.join(DATA_ROOT, "Occluded_Duke")
        print(f"[data] occ_duke       <- DukeMTMC-reID + official lists: {msg}")
    except Exception as e:  # noqa: BLE001
        print(f"!! Occluded-Duke could not be built: {type(e).__name__}: {e}")

# ---- Occluded-REID: official zip -> repo layout ------------------------------------------------------------------------
OCCREID_URL = ("https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/raw/master/"
               "Occluded_REID.zip")
OCCREID_SHA256 = "1c8854e02db775da6a51399df4b7e02c299804850dff9985451d610fc8c6d57d"
OCCREID_SPLIT_SEED = 0


def build_occluded_reid(out_dir):
    if osp.isdir(osp.join(out_dir, "test", "gallery")) and len(os.listdir(osp.join(out_dir, "test", "gallery"))) == 100:
        return "already built"
    zpath = next((osp.join(d, "Occluded_REID.zip") for d in ALL_INPUT_DIRS if osp.isfile(osp.join(d, "Occluded_REID.zip"))), None)
    zpath = zpath or _download(OCCREID_URL, osp.join(CACHE_DIR, "Occluded_REID.zip"), OCCREID_SHA256)
    tmp = out_dir + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff", ".jpg", ".png"))]
        pids = sorted({n.split("/")[-2] for n in names})
        rng = random.Random(OCCREID_SPLIT_SEED)
        train_ids = set(rng.sample(pids, len(pids) // 2))
        for n in names:
            kind, pid, fname = n.split("/")[-3:]
            occluded = kind.startswith("occluded")
            if pid in train_ids:
                sub = osp.join("train", pid)
            else:
                sub = osp.join("test", "query" if occluded else "gallery", pid)
            os.makedirs(osp.join(tmp, sub), exist_ok=True)
            stem = ("occ_" if occluded else "whole_") + osp.splitext(fname)[0]
            with z.open(n) as f:
                Image.open(f).convert("RGB").save(osp.join(tmp, sub, stem + ".jpg"), quality=95)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return f"built ({len(pids)} IDs: {len(train_ids)} train / {len(pids) - len(train_ids)} test, seed {OCCREID_SPLIT_SEED})"


if "occ_reid" in DATASETS_TO_RUN:
    try:
        msg = build_occluded_reid(osp.join(DATA_ROOT, "Occluded_REID"))
        PREPARED["occ_reid"] = osp.join(DATA_ROOT, "Occluded_REID")
        print(f"[data] occ_reid       <- official zip: {msg}")
    except Exception as e:  # noqa: BLE001
        print(f"!! Occluded-REID could not be prepared: {type(e).__name__}: {e}")

missing = [d for d in DATASETS_TO_RUN if d not in PREPARED]
print("\nprepared:", sorted(k for k in PREPARED if k != "dukemtmcreid"))
if missing:
    print(f"!! not available, will be skipped: {missing}")
