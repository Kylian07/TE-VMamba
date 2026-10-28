# %% [markdown]
# ## Datasets, sampler, transforms (port of `datasets/*.py`)
# * Market-1501 / DukeMTMC / Occluded-Duke: filenames `PPPP_cC...jpg` → (pid, camera). Train IDs are relabelled to 0..N−1.
# * Occluded-REID: one folder per identity. **Query = camera 0, gallery = camera 1**, so the standard evaluator
#   (which ignores same-ID *and* same-camera gallery hits) keeps every true match.
# * PK sampler: 16 identities × 4 images = batch 64.
# * Train transform: Resize 224×224 (bicubic), flip 0.5, pad 10 + random crop, ImageNet normalisation, random erasing
#   0.3 (pixel mode).
#
# Only change from the repo: the `PPPP_cC` pattern is applied to the file name, not the full path (identical results;
# avoids accidental matches on digits in Kaggle paths).

# %%
def read_image(img_path):
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    return Image.open(img_path).convert("RGB")


class BaseImageDataset:
    def get_imagedata_info(self, data):
        pids = {d[1] for d in data}; cams = {d[2] for d in data}; tracks = {d[3] for d in data}
        return len(pids), len(data), len(cams), len(tracks)

    def print_dataset_statistics(self, train, query, gallery, tag):
        print(f"=> {tag} loaded\n  subset   | # ids | # images | # cameras")
        for n, d in (("train", train), ("query", query), ("gallery", gallery)):
            p, i, c, _ = self.get_imagedata_info(d)
            print(f"  {n:8s} | {p:5d} | {i:8d} | {c:9d}")

    def _finish(self, train, query, gallery, tag, verbose):
        self.train, self.query, self.gallery = train, query, gallery
        if verbose:
            self.print_dataset_statistics(train, query, gallery, tag)
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(train)


class _MarketLike(BaseImageDataset):
    """market1501.py / dukemtmcreid.py / occ_duke.py share this parser."""
    dataset_dir, tag, max_cam, skip_junk = "", "", 8, False

    def __init__(self, root="", verbose=True, pid_begin=0, **kw):
        self.pid_begin = pid_begin
        d = osp.join(root, self.dataset_dir)
        tr = self._process_dir(osp.join(d, "bounding_box_train"), relabel=True)
        q = self._process_dir(osp.join(d, "query"), relabel=False)
        g = self._process_dir(osp.join(d, "bounding_box_test"), relabel=False)
        self._finish(tr, q, g, self.tag, verbose)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d)")
        parsed = [(p, *map(int, pattern.search(osp.basename(p)).groups())) for p in img_paths]
        if self.skip_junk:
            parsed = [x for x in parsed if x[1] != -1]
        pid2label = {pid: lab for lab, pid in enumerate(sorted({x[1] for x in parsed}))}
        out = []
        for p, pid, camid in parsed:
            assert 1 <= camid <= self.max_cam
            out.append((p, self.pid_begin + (pid2label[pid] if relabel else pid), camid - 1, 1))
        return out


class Market1501(_MarketLike):
    dataset_dir, tag, max_cam, skip_junk = "market1501", "Market-1501", 6, True


class DukeMTMCreID(_MarketLike):
    dataset_dir, tag = "dukemtmcreid", "DukeMTMC-reID"


class OCC_DukeMTMCreID(_MarketLike):
    dataset_dir, tag = "Occluded_Duke", "Occluded-DukeMTMC"


class OCC_ReID(BaseImageDataset):
    dataset_dir = "Occluded_REID"

    def __init__(self, root="", verbose=True, pid_begin=0, **kw):
        d = osp.join(root, self.dataset_dir)
        tr = self._process(osp.join(d, "train"), "train")
        q = self._process(osp.join(d, "test/query"), "query")
        g = self._process(osp.join(d, "test/gallery"), "gallery")
        self._finish(tr, q, g, "Occluded-ReID", verbose)

    def _process(self, folder, mode):
        pid_dirs = sorted(os.listdir(folder))
        pid2label = {pid: idx for idx, pid in enumerate(pid_dirs)}
        out = []
        for pid in pid_dirs:
            if not osp.isdir(osp.join(folder, pid)):
                continue
            camid = 1 if mode == "gallery" else 0
            for img in sorted(glob.glob(osp.join(folder, pid, "*.jpg"))):
                out.append((img, pid2label[pid], camid, 1))
        return out


FACTORY = {"market1501": Market1501, "dukemtmc": DukeMTMCreID, "occ_duke": OCC_DukeMTMCreID, "occ_reid": OCC_ReID}


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, trackid, img_path.split("/")[-1]


class RandomIdentitySampler(Sampler):
    """N identities x K instances per batch (verbatim logic of datasets/sampler.py)."""
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source, self.batch_size, self.num_instances = data_source, batch_size, num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = max(len(self.index_dic[pid]), num_instances)
            self.length += num - num % num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch = []
            for idx in idxs:
                batch.append(idx)
                if len(batch) == self.num_instances:
                    batch_idxs_dict[pid].append(batch); batch = []
        avai_pids, final_idxs = copy.deepcopy(self.pids), []
        while len(avai_pids) >= self.num_pids_per_batch:
            for pid in random.sample(avai_pids, self.num_pids_per_batch):
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    return (torch.stack(imgs, 0), torch.tensor(pids, dtype=torch.int64), torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64))


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    return (torch.stack(imgs, 0), pids, camids, torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64), img_paths)


def _quick_eval_subset(query, gallery, seed=0):
    rng = random.Random(seed)
    q = rng.sample(query, min(len(query), QUICK["eval_max_query"]))
    qp = {x[1] for x in q}
    pos = [x for x in gallery if x[1] in qp]; neg = [x for x in gallery if x[1] not in qp]
    return q, pos + rng.sample(neg, max(0, min(len(neg), QUICK["eval_max_gallery"] - len(pos))))


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=T.InterpolationMode.BICUBIC),   # == interpolation=3
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
    ])
    val_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    dataset = FACTORY[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    query, gallery = dataset.query, dataset.gallery
    if RUN_MODE == "quick":
        query, gallery = _quick_eval_subset(query, gallery)
    nw = cfg.DATALOADER.NUM_WORKERS
    train_loader = DataLoader(ImageDataset(dataset.train, train_transforms), batch_size=cfg.SOLVER.IMS_PER_BATCH,
                              sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                              num_workers=nw, collate_fn=train_collate_fn, pin_memory=True, persistent_workers=nw > 0)
    val_loader = DataLoader(ImageDataset(query + gallery, val_transforms), batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=False, num_workers=nw, collate_fn=val_collate_fn, pin_memory=True)
    return train_loader, val_loader, len(query), dataset.num_train_pids, dataset
