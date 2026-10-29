# %% [markdown]
# ## Evaluation (port of `utils/metrics.py`)
# Features are L2-normalised, the distance is cosine distance (1 − q·g), and gallery images with the same ID **and**
# the same camera as the query are ignored. CMC (Rank-1/5/10) and mAP follow the standard Market-1501 protocol.
# Re-ranking is off, as in the repo (the evaluator is created without it). Features are extracted in fp32, as in the repo.

# %%
def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP = [], []
    for q_idx in range(num_q):
        order = indices[q_idx]
        keep = np.invert((g_pids[order] == q_pids[q_idx]) & (g_camids[order] == q_camids[q_idx]))
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        tmp = orig_cmc.cumsum() / (np.arange(1, orig_cmc.shape[0] + 1) * 1.0)
        all_AP.append((tmp * orig_cmc).sum() / orig_cmc.sum())
    assert all_cmc, "Error: all query identities do not appear in gallery"
    return np.asarray(all_cmc).astype(np.float32).mean(0), float(np.mean(all_AP))


@torch.no_grad()
def evaluate(model, val_loader, num_query, feat_norm=True):
    model.eval()
    feats, pids, camids = [], [], []
    for img, pid, camid, camids_b, _, _ in val_loader:
        f = model(img.to(DEVICE, non_blocking=True))          # fp32, like processor.py's evaluation
        feats.append(f.float().cpu()); pids.extend(int(x) for x in pid); camids.extend(int(x) for x in camid)
    feats = torch.cat(feats)
    if feat_norm:
        feats = F.normalize(feats, dim=1, p=2)
    qf, gf = feats[:num_query], feats[num_query:]
    distmat = (1.0 - qf @ gf.t()).numpy() if feat_norm else torch.cdist(qf, gf).numpy()
    p, c = np.asarray(pids), np.asarray(camids)
    cmc, mAP = eval_func(distmat, p[:num_query], p[num_query:], c[:num_query], c[num_query:])
    return {"mAP": mAP, "R1": float(cmc[0]), "R5": float(cmc[4]), "R10": float(cmc[9])}


# %% [markdown]
# ## Efficiency measurement (what the paper's "over 60% FLOPs reduction" refers to)
# With the trained model, the notebook measures:
# * **tokens processed per block** (via forward hooks on every VSS block) with token adaptation on vs off;
# * **inference throughput** (images/s, batch 64, fp16) with token adaptation on vs off;
# * the average **keep ratios** of the prune modules and of the merge modules.
#
# FLOPs are not counted directly (the repo's FLOP counter needs fvcore plus custom op handlers). The reduction in
# tokens per block is a close proxy, because each block's cost grows linearly with its token count.

# %%
@torch.no_grad()
def measure_efficiency(model, size, batch=64, iters=20):
    model.eval()
    x = torch.randn(batch, 3, size[0], size[1], device=DEVICE)
    tokens, chans = [], []

    def hook(_m, inp, _out):
        t = inp[0]                       # (B, C, H, W) for the ln2d (channel-first) configs
        tokens.append(t.shape[-1] * t.shape[-2] if t.dim() == 4 else t.shape[1])
        chans.append(t.shape[1] if t.dim() == 4 else t.shape[-1])
    blocks = [m for m in model.modules() if m.__class__.__name__ == "VSSBlock"]
    res = {}
    for flag in (True, False):
        vmamba_mod.ENABLE_TOKEN_ADAPTATION = flag
        handles = [b.register_forward_hook(hook) for b in blocks]
        tokens.clear(); chans.clear()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            model(x)
        for h in handles:
            h.remove()
        per_block = list(tokens)
        cost = float(sum(n * c * c for n, c in zip(tokens, chans)))   # per-block cost ~ tokens x dim^2
        pr, mr = token_stats(model) if flag else ([], [])
        n = iters if DEVICE == "cuda" else 2
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        res["on" if flag else "off"] = {"tokens_per_block": per_block, "total_tokens": int(sum(per_block)), "cost_proxy": cost,
                                        "img_per_s": batch * n / (time.time() - t0),
                                        "prune_keep": pr, "merge_keep": mr}
    vmamba_mod.ENABLE_TOKEN_ADAPTATION = True
    res["token_reduction_%"] = 100.0 * (1 - res["on"]["total_tokens"] / max(res["off"]["total_tokens"], 1))
    # FLOPs proxy: the linear layers (in/out projections, MLP) dominate a VSS block and scale with tokens x dim^2,
    # so this weights each block's token count by its width. It is comparable to the paper's GFLOPs ratio.
    res["flops_proxy_reduction_%"] = 100.0 * (1 - res["on"]["cost_proxy"] / max(res["off"]["cost_proxy"], 1e-9))
    res["speedup_x"] = res["on"]["img_per_s"] / max(res["off"]["img_per_s"], 1e-9)
    return res


# %% [markdown]
# ## Training loop (port of `processor/processor.py::do_train` + `train.py`)
# Per epoch: `scheduler.step(epoch)`, then for each PK batch run an AMP forward, compute ID + triplet loss, scale,
# backward and step. Seed 1234, `cudnn.deterministic=True`, `benchmark=True`, as in `train.py`.
#
# `train.py` runs **one forward pass before building the optimiser** (a debug check). That pass is what lazily
# creates each merge module's threshold MLP. It is reproduced here, so those parameters exist when the optimiser and
# checkpoints are built.
#
# Additions for Kaggle (no effect on the maths):
# * a resumable state (model, optimiser, AMP scaler, epoch, RNG states) saved every `CKPT_EVERY` epochs;
# * a clean stop before `MAX_SESSION_HOURS`;
# * optional mid-training evaluations (`EVAL_EVERY`).

# %%
T_NOTEBOOK_START = globals().get("T_NOTEBOOK_START", time.time())


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_logger(out_dir):
    lg = logging.getLogger("tev"); lg.setLevel(logging.INFO); lg.handlers.clear(); lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(osp.join(out_dir, "train_log.txt"), mode="a")):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg


def _state_path(name):
    return osp.join(OUT_DIR, name, "last_state.pth")


def _find_resume(name):
    p = _state_path(name)
    if osp.isfile(p):
        return p
    for q in glob.glob(osp.join(INPUT_ROOT, "**", "TEVMamba_outputs", name, "last_state.pth"), recursive=True):
        os.makedirs(osp.dirname(p), exist_ok=True); shutil.copy(q, p)
        for f in ("train_log.txt", "eval_history.json"):
            if osp.isfile(osp.join(osp.dirname(q), f)):
                shutil.copy(osp.join(osp.dirname(q), f), osp.join(osp.dirname(p), f))
        print(f"[resume] copied {q}")
        return p
    return None


def _probe_batch_memory(model, cfg, num_classes, loss_fn):
    """PROTOCOL="paper" uses batch 128. Try one AMP forward+backward at that size on GPU 0. If it does not fit and a
    second GPU is visible, train with nn.DataParallel: each GPU runs half the batch, but the outputs are gathered on
    GPU 0 before the loss, so the batch-hard triplet loss still mines over all 128 images. The probe leaves the
    weights, BN statistics and RNG state exactly as they were."""
    if DEVICE != "cuda" or cfg.SOLVER.IMS_PER_BATCH <= 64:
        return model
    bs, (h, w) = cfg.SOLVER.IMS_PER_BATCH, cfg.INPUT.SIZE_TRAIN
    saved = copy.deepcopy(model.state_dict())
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
    ok = True
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(bs, 3, h, w, device=DEVICE)
        y = torch.arange(bs, device=DEVICE) % num_classes
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            score, feat = model(x, y)
            loss = loss_fn(score, feat, y)
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"[memory] batch {bs} fits on one GPU: peak {peak:.1f} GiB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB")
    except torch.cuda.OutOfMemoryError:
        ok = False
    del x
    model.zero_grad(set_to_none=True)
    model.load_state_dict(saved)
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    torch.cuda.empty_cache()
    if ok:
        return model
    if torch.cuda.device_count() >= 2:
        print(f"[memory] batch {bs} does not fit on one GPU -> nn.DataParallel over {torch.cuda.device_count()} GPUs "
              "(loss still computed over the full batch)")
        return nn.DataParallel(model)
    raise RuntimeError(f"batch {bs} does not fit on one GPU and no second GPU is visible: choose 'GPU T4 x2'")


def train_and_eval(name):
    cfg = build_cfg(name)
    key = run_key(name)
    out_dir = osp.join(OUT_DIR, key); os.makedirs(out_dir, exist_ok=True)
    final_path = osp.join(out_dir, "final_results.json")
    if osp.isfile(final_path) and RESUME:
        print(f"[skip] {key}: already finished -> {final_path}")
        return json.load(open(final_path))
    lg = setup_logger(out_dir)
    set_seed(cfg.SOLVER.SEED)
    train_loader, val_loader, num_query, num_classes, dataset = make_dataloader(cfg)
    model = VMambaReID(num_classes, cfg, resolve_pretrain_path()).to(DEVICE)
    lg.info(f"[{key}] protocol={PROTOCOL} | batch={cfg.SOLVER.IMS_PER_BATCH} | "
            f"label smoothing={cfg.MODEL.IF_LABELSMOOTH} | prune={list(cfg.MODEL.TOKEN_ADAPT.PRUNE_LAYERS)} | "
            f"merge={list(cfg.MODEL.TOKEN_ADAPT.MERGE_LAYERS)}")
    lg.info(f"[{key}] classes={num_classes} | epochs={cfg.SOLVER.MAX_EPOCHS} | input={cfg.INPUT.SIZE_TRAIN} | "
            f"backbone={MODEL_SIZE} | {model.pretrain_report}")

    # == train.py: one forward in eval mode before the optimiser (creates the lazy merge-gate MLPs)
    model.eval()
    with torch.no_grad():
        model(next(iter(train_loader))[0][:8].to(DEVICE))
    pr, mr = token_stats(model)
    lg.info(f"[attach] prune modules={len(pr)} keep~{np.mean(pr) if pr else float('nan'):.2f} | "
            f"merge modules={len(mr)} keep~{np.mean(mr) if mr else float('nan'):.2f}")
    model.train()

    loss_fn = make_loss(cfg, num_classes)
    train_model = _probe_batch_memory(model, cfg, num_classes, loss_fn)   # == model unless batch 128 needs 2 GPUs
    model.train()
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    start_epoch, history = 1, []
    rp = _find_resume(key) if RESUME else None
    if rp:
        st = torch.load(rp, map_location="cpu", weights_only=False)
        # strict=False like the repo's test.py: VSSM._load_from_state_dict renames "head.*" -> "classifier.head.*"
        # (legacy-checkpoint compatibility), so the unused 0-output `base.head` cannot round-trip strictly.
        inc = model.load_state_dict(st["model"], strict=False)
        bad = [k for k in inc.missing_keys + inc.unexpected_keys if not k.startswith(("base.head.", "base.classifier.head."))]
        assert not bad, f"resume state does not match the model: {bad[:10]}"
        optimizer.load_state_dict(st["optimizer"]); scaler.load_state_dict(st["scaler"])
        start_epoch, history = st["epoch"] + 1, st.get("history", [])
        random.setstate(st["rng_py"]); np.random.set_state(st["rng_np"]); torch.set_rng_state(st["rng_torch"])
        if DEVICE == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"])
        lg.info(f"[resume] from epoch {st['epoch']} ({rp})")

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history, "rng_py": random.getstate(), "rng_np": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if DEVICE == "cuda" else None}, _state_path(key) + ".tmp")
        os.replace(_state_path(key) + ".tmp", _state_path(key))

    epochs = cfg.SOLVER.MAX_EPOCHS
    epoch_times = []
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        scheduler.step(epoch)
        model.train()
        loss_sum = acc_sum = n_seen = 0.0
        n_iter = 0
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            if RUN_MODE == "quick" and n_iter >= QUICK["max_iters"]:
                break
            optimizer.zero_grad(set_to_none=True)
            img, target = img.to(DEVICE, non_blocking=True), vid.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                score, feat = train_model(img, target)
                loss = loss_fn(score, feat, target, target_cam)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            acc = (score.max(1)[1] == target).float().mean().item()
            loss_sum += loss.item() * img.shape[0]; acc_sum += acc; n_seen += img.shape[0]
            if (n_iter + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                lg.info(f"Epoch[{epoch}] Iter[{n_iter + 1}/{len(train_loader)}] Loss {loss_sum / n_seen:.3f} "
                        f"Acc {acc_sum / (n_iter + 1):.3f} Lr {optimizer.param_groups[0]['lr']:.2e}")
        dt = time.time() - t0
        epoch_times.append(dt)
        lg.info(f"Epoch {epoch}/{epochs} done: loss {loss_sum / max(n_seen, 1):.3f} acc {acc_sum / max(n_iter + 1, 1):.3f} "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | {dt:.0f}s ({n_seen / max(dt, 1e-9):.0f} img/s)")
        if epoch in (start_epoch, start_epoch + 5):
            # the first epoch includes cuDNN/Triton warm-up (~2x slower), so the estimate is refined 5 epochs later
            remaining = np.mean(epoch_times[-3:]) * (epochs - epoch) / 3600
            left = MAX_SESSION_HOURS - (time.time() - T_NOTEBOOK_START) / 3600
            lg.info(f"[estimate] ~{remaining:.1f} h of training left for {key} at this speed, {left:.1f} h left in this "
                    f"session ({'fits' if remaining < left else 'needs resume across sessions'})"
                    + (f" | first epoch includes warm-up; refined at epoch {start_epoch + 5}" if epoch == start_epoch else ""))
        if EVAL_EVERY and epoch % EVAL_EVERY == 0 and epoch != epochs:
            r = evaluate(model, val_loader, num_query); r["epoch"] = epoch; history.append(r)
            lg.info(f"[eval] epoch {epoch}: mAP {r['mAP']:.1%} R1 {r['R1']:.1%}")
        if epoch % CKPT_EVERY == 0 or epoch == epochs:
            save_state(epoch)
        elapsed_h = (time.time() - T_NOTEBOOK_START) / 3600
        if epoch != epochs and elapsed_h + 1.5 * np.mean(epoch_times) / 3600 + 0.25 > MAX_SESSION_HOURS:
            save_state(epoch)
            lg.info(f"[time] {elapsed_h:.1f} h used: stopping {key} at epoch {epoch}. Attach this output and re-run to resume.")
            return None

    r = evaluate(model, val_loader, num_query); r["epoch"] = epochs; history.append(r)
    lg.info(f"[final] {key}: mAP {r['mAP']:.1%} | R1 {r['R1']:.1%} | R5 {r['R5']:.1%} | R10 {r['R10']:.1%}")
    vmamba_mod.ENABLE_TOKEN_ADAPTATION = False
    r_noadapt = evaluate(model, val_loader, num_query)
    vmamba_mod.ENABLE_TOKEN_ADAPTATION = True
    lg.info(f"[diagnostic] same weights, token pruning/merging switched OFF at test: mAP {r_noadapt['mAP']:.1%} R1 {r_noadapt['R1']:.1%}")
    eff = measure_efficiency(model, cfg.INPUT.SIZE_TEST)
    lg.info(f"[efficiency] tokens/forward -{eff['token_reduction_%']:.1f}% | FLOPs proxy -{eff['flops_proxy_reduction_%']:.1f}% | throughput on {eff['on']['img_per_s']:.0f} "
            f"vs off {eff['off']['img_per_s']:.0f} img/s (x{eff['speedup_x']:.2f})")
    torch.save(model.state_dict(), osp.join(out_dir, "final_model.pth"))
    result = {"dataset": name, "protocol": PROTOCOL, "batch": cfg.SOLVER.IMS_PER_BATCH, "model_size": MODEL_SIZE, "pretrain": model.pretrain_report, "epochs": epochs,
              "run_mode": RUN_MODE, "final": r, "token_adapt_off_at_test": r_noadapt, "efficiency": eff,
              "history": history}
    json.dump(result, open(final_path, "w"), indent=1)
    if osp.isfile(_state_path(key)):
        os.remove(_state_path(key))            # finished: the resumable state is no longer needed
    return result
