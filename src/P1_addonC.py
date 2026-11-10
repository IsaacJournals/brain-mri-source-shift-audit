# P1 ADD-ON C (approved 8 Nov 2026, Step 5b upgrade package): items 1, 2, 6.
#  C1 independent external test sets (BDNeuro-MRI, PMRAM) + lineage/containment check against the Kaggle family (lesson L2)
#  C2 SourceShield: brain-masked inputs + Group DRO over origin sources, vs ERM / harmonised / mask-only / DRO-only
#  C3 frozen DINOv2 ViT-S/14 + logistic-regression probe on the same protocols and external sets
# Executed inside P1_pipeline.py's namespace (P1_ADDON=1). Per-epoch checkpoints: a disconnect loses at most one epoch.
import io, zipfile, json as _json, urllib.request, shutil as _sh
import scipy.stats as _st
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression

AO = OUT / "addonC"; [(AO / d).mkdir(parents=True, exist_ok=True) for d in ("tables", "figures", "checkpoints")]
STATUS = ROOT / "ADDON_STATUS.md"
def cstatus(msg):
    line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | C | {msg}"; print(line)
    with open(STATUS, "a") as f: f.write(line + "\n")
if not LOCAL_DUMMY and not (ROOT / "P1_results_FULL_PAPER_RUN.zip").exists():
    raise RuntimeError("FULL_PAPER_RUN not finished. Fix: wait for v7 CHAIN COMPLETE; run one GPU notebook at a time (L14).")
cstatus("ADD-ON C START")
CLS4 = TYPES + ["notumor"]
SEEDS_C = P["seeds"]; ARCHS_C = [a for a in P["archs"] if a != "convnext_tiny"] or P["archs"][:1]
EPOCHS_C = P["epochs"]

# ======================= C1: external datasets =======================
EXT = {"bdneuro": ("zwr4ntf94j", [8, 7, 6, 5]), "pmram": ("m7w55sw88b", [1])}
def mendeley_fetch(name, did, versions):
    dest = DATA / f"ext_{name}"
    if (dest / ".complete").exists(): return dest
    dest.mkdir(parents=True, exist_ok=True)
    files = None
    def _getjson(u):
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 research-notebook", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r: return _json.loads(r.read())
    try:
        meta = _getjson(f"https://data.mendeley.com/public-api/datasets/{did}")
        cur = int(meta.get("current_version")); versions = [cur] + [v for v in versions if v != cur]
    except Exception as e: print("mendeley meta failed", name, type(e).__name__)
    for v in versions:
        for k in range(3):
            try:
                files = _getjson(f"https://data.mendeley.com/public-api/datasets/{did}/files?folder_id=root&version={v}")
                if files: break
            except Exception as e: print("mendeley list failed", name, v, type(e).__name__, getattr(e, "code", "")); time.sleep(5)
        if files:
            (dest / "VERSION_USED.txt").write_text(f"10.17632/{did}.{v}\n"); print("mendeley", name, "version", v); break
    if not files: raise RuntimeError(f"Could not list Mendeley files for {name}. Fix: re-run; if it persists, download the zip from data.mendeley.com/datasets/{did} into MyDrive/MI10_P1/ext_{name}/")
    def walk(items):
        for it in items:
            if "content_details" in it and it["content_details"].get("download_url"):
                yield it["filename"], it["content_details"]["download_url"]
    for fn, url in walk(files):
        if not fn.lower().endswith((".zip", ".rar", ".jpg", ".png", ".jpeg")): continue
        if name == "pmram" and "aug" in fn.lower(): continue                   # raw images only
        out = dest / fn; http_get(url, out)
        if fn.lower().endswith(".zip"):
            with zipfile.ZipFile(out) as z: z.extractall(dest)
            out.unlink()
    (dest / ".complete").write_text("ok"); return dest

def ext_manifest(name, root):
    rows = []
    for fp in sorted(root.rglob("*")):
        if fp.suffix.lower() not in IMG_EXT: continue
        rel = [p.lower() for p in fp.relative_to(root).parts]
        if name == "pmram" and any(("aug" in p) for p in rel): continue            # raw images only
        lab = infer_label(fp.relative_to(root).parts)
        if lab is None or lab == "tumor_unspec": continue
        rows.append({"dataset": name, "path": str(fp), "label": lab})
    return pd.DataFrame(rows)

if LOCAL_DUMMY:   # schema-matched dummy external set for the local plumbing test
    rows = []
    for name in ("bdneuro", "pmram"):
        d = DATA / f"ext_{name}"
        for lab in CLS4:
            (d / lab).mkdir(parents=True, exist_ok=True)
            for i in range(6):
                p = d / lab / f"{lab}_{i}.png"
                if not p.exists():
                    a = (np.random.default_rng(i + 7 * len(lab)).random((96, 96)) * 255).astype(np.uint8); Image.fromarray(a).save(p)
                rows.append({"dataset": name, "path": str(p), "label": lab})
    em = pd.DataFrame(rows)
else:
    em = pd.concat([ext_manifest(n, mendeley_fetch(n, d, v)) for n, (d, v) in EXT.items()], ignore_index=True)
if em.empty: raise RuntimeError("No external images found after download. Fix: check folder names in the downloaded zips.")
cstatus("external images: " + em.groupby(["dataset", "label"]).size().to_string().replace("\n", "; "))

def _ext_one(p):
    a = load_gray(p); return raw_repr(a), harm_repr(a), dedup_vec(a)
E_raw = np.zeros((len(em), IMG, IMG), np.uint8); E_harm = np.zeros_like(E_raw); EV = np.zeros((len(em), DEDUP_SIZE * DEDUP_SIZE), np.float32)
for i, p in enumerate(em.path):
    E_raw[i], E_harm[i], EV[i] = _ext_one(p)

# lineage check (L2): max NCC (8 dihedral transforms) of each external image to any Kaggle-family image
dev = DEVICE; dt = torch.float16 if dev == "cuda" else torch.float32
Vt = torch.tensor(V, dtype=dt, device=dev); emax = []
for s in range(0, len(EV), 1024):
    q = torch.tensor(EV[s:s + 1024], dtype=dt, device=dev); m = None
    for qt in dihedral(q):
        sim = (qt @ Vt.T).max(1).values; m = sim if m is None else torch.maximum(m, sim)
    emax.append(m.float().cpu().numpy())
em["max_ncc_kaggle"] = np.concatenate(emax); thr = GATES["NCC_THRESHOLD"]
em["contained"] = em.max_ncc_kaggle >= thr
# within-external near-duplicates (e.g. leftover augmented copies) -> groups for bootstrap
eu = UF(len(em)); ep, _ = near_pairs(EV, thr)
for a_, b_, v_ in ep:
    if v_ >= thr: eu.u(a_, b_)
em["ext_group"] = [eu.f(i) for i in range(len(em))]
cont = em.groupby("dataset").agg(n=("path", "size"), contained_pct=("contained", lambda s: round(100 * s.mean(), 2)),
                                 n_groups=("ext_group", "nunique")).reset_index()
save_csv(cont, AO / "tables" / "external_containment.csv"); save_csv(em.drop(columns=["path"]), AO / "tables" / "external_manifest.csv")
cstatus("containment vs Kaggle family: " + "; ".join(f"{r.dataset} {r.contained_pct}% of {r.n} (groups {r.n_groups})" for r in cont.itertuples()))
INDEP = {r.dataset: r.contained_pct <= 5.0 for r in cont.itertuples()}
ekeep = ~em.contained.values                                            # only images with no near-twin in the Kaggle family

# ======================= helpers =======================
def masked(Xu8, ids=None):
    A = Xu8 if ids is None else Xu8[ids]
    out = A.copy()
    for k in range(len(out)): out[k][~brain_mask(out[k])] = 0
    return out
def tens(A):
    x = torch.from_numpy(A).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1); return (x - MEAN) / STD
def predict_A(model, A):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        for s in range(0, len(A), 128): out.append(torch.softmax(model(tens(A[s:s + 128]).to(DEVICE)).float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))
def boot_ba(y, pred, groups, B=1000, seed=0):
    r = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in r.choice(ug, len(ug))]); vals.append(balanced_accuracy_score(y[take], pred[take]))
    return np.percentile(vals, [2.5, 97.5])
def ext_eval(prob_fn, rep, task, tag_rows):
    """Evaluate on each external dataset (clean images only), label space mapped to the task."""
    rows = []
    for ds in em.dataset.unique():
        sel = (em.dataset.values == ds) & ekeep
        lab = em.label.values
        if task == "four": keep = sel & np.isin(lab, CLS4); y = np.array([CLS4.index(l) for l in lab[keep]])
        elif task == "type": keep = sel & np.isin(lab, TYPES); y = np.array([TYPES.index(l) for l in lab[keep]])
        else: keep = sel; y = (lab[keep] != "notumor").astype(int)
        if keep.sum() == 0 or len(np.unique(y)) < 2: continue
        A = {"raw": E_raw, "harm": E_harm, "mask": None}[rep]
        A = masked(E_raw, np.where(keep)[0]) if rep == "mask" else A[keep]
        prob = prob_fn(A); r = metrics(y, prob, prob.shape[1]); lo, hi = boot_ba(y, prob.argmax(1), em.ext_group.values[keep])
        rows.append({**tag_rows, "test": f"external_{ds}", "independent": INDEP.get(ds, False), **r, "ba_ci_lo": lo, "ba_ci_hi": hi})
    return rows

# ======================= C1a: existing FULL models on the external sets =======================
def load_full(pr, arch, seed):
    ver = "p1-v1.3cnx" if arch.startswith("convnext") else MODEL_VERSION
    ck = SUB["checkpoints"] / f"{pr['name']}_{arch}_s{seed}_{MODE}_{ver}_{SPLIT_SEED}.pt"
    if not ck.exists(): return None
    try: st_ = torch.load(ck, map_location=DEVICE)
    except Exception: st_ = torch.load(ck, map_location=DEVICE, weights_only=False)
    m = timm.create_model(arch, pretrained=False, num_classes=NCLS[pr["task"]]).to(DEVICE); m.load_state_dict(st_["model"]); m.eval(); return m
c1 = []
_c1f = AO / "tables" / "external_eval_full_models.csv"
for pr in ([] if (_c1f.exists() and not LOCAL_DUMMY) else [p for p in PROTOCOLS if p["split"] in ("group", "official")]):
    for arch in P["archs"]:
        for seed in SEEDS_C:
            m = load_full(pr, arch, seed)
            if m is None: continue
            c1 += ext_eval(lambda A: predict_A(m, A), pr["rep"], pr["task"], {"model": "ERM_" + pr["rep"], "protocol": pr["name"], "arch": arch, "seed": seed})
            del m; torch.cuda.empty_cache()
    cstatus(f"C1 external eval done {pr['name']}")
if c1: C1 = pd.DataFrame(c1); save_csv(C1, _c1f)
else: C1 = pd.read_csv(_c1f); cstatus("C1 resume (loaded external_eval_full_models.csv)")

# ======================= C3: frozen DINOv2 linear probe =======================
_c3f = AO / "tables" / "foundation_model_probe.csv"
if _c3f.exists() and not LOCAL_DUMMY:
    C3 = pd.read_csv(_c3f); fm_name = str(C3.arch.iloc[0]); cstatus("C3 resume (loaded foundation_model_probe.csv)")
else:
    def dino_feats(A_list):
        try: fm = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=not LOCAL_DUMMY, num_classes=0, img_size=IMG).to(DEVICE).eval()
        except Exception as e:
            print("DINOv2 unavailable:", type(e).__name__, "-> fallback resnet50 ImageNet features (labelled)"); fm = timm.create_model("resnet50", pretrained=not LOCAL_DUMMY, num_classes=0).to(DEVICE).eval()
        outs = []
        for A in A_list:
            f = []
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                for s in range(0, len(A), 128): f.append(fm(tens(A[s:s + 128]).to(DEVICE)).float().cpu().numpy())
            outs.append(np.concatenate(f) if f else np.zeros((0, 384)))
        return outs, fm.default_cfg.get("architecture", "dinov2")
    (F_raw, F_ext), fm_name = dino_feats([X_raw, E_raw])
    c3 = []
    for pr in [p for p in PROTOCOLS if p["rep"] == "raw" and p["split"] in ("group", "official")]:
        for seed in SEEDS_C:
            m_, tr, va, tests = make_split(pr, seed); y_all = np.full(len(man), -1); y_all[m_.img_id.values] = m_.y.values
            for k_ in tests:
                if k_.startswith("cross_") and isinstance(tests[k_], np.ndarray):
                    mb = task_frame(pr["task"], k_.replace("cross_", "")); y_all[mb.img_id.values] = mb.y.values
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(F_raw[tr], y_all[tr])
            for tname, idx in tests.items():
                if not isinstance(idx, np.ndarray) or not len(idx): continue
                c3.append({"model": f"{fm_name}_frozen_LR", "protocol": pr["name"], "arch": fm_name, "seed": seed, "test": tname,
                           **metrics(y_all[idx], clf.predict_proba(F_raw[idx]), NCLS[pr["task"]])})
            # external: rebuild the same selection as ext_eval but on features
            for ds in em.dataset.unique():
                sel = (em.dataset.values == ds) & ekeep; lab = em.label.values
                if pr["task"] == "four": keep = sel & np.isin(lab, CLS4); y = np.array([CLS4.index(l) for l in lab[keep]])
                elif pr["task"] == "type": keep = sel & np.isin(lab, TYPES); y = np.array([TYPES.index(l) for l in lab[keep]])
                else: keep = sel; y = (lab[keep] != "notumor").astype(int)
                if keep.sum() == 0 or len(np.unique(y)) < 2: continue
                prob = clf.predict_proba(F_ext[keep]); r = metrics(y, prob, prob.shape[1]); lo, hi = boot_ba(y, prob.argmax(1), em.ext_group.values[keep])
                c3.append({"model": f"{fm_name}_frozen_LR", "protocol": pr["name"], "arch": fm_name, "seed": seed, "test": f"external_{ds}",
                           "independent": INDEP.get(ds, False), **r, "ba_ci_lo": lo, "ba_ci_hi": hi})
    C3 = pd.DataFrame(c3); save_csv(C3, AO / "tables" / "foundation_model_probe.csv"); cstatus(f"C3 foundation-model probe done ({fm_name})")

# ======================= C2: SourceShield =======================
# Training data: Nickparvar cluster split (4-class). Group = origin source of each image's duplicate cluster
# (figshare / sartaj / br35h / nickparvar_only). Tested on the in-source cluster test and the external sets.
src_by_group = man.groupby("group").source.apply(lambda s: set(s))
def origin(i):
    s = src_by_group[man.group.iloc[i]]
    for o in ("figshare", "sartaj", "br35h"):
        if o in s: return o
    return "nickparvar_only"
ORIGINS = ["figshare", "sartaj", "br35h", "nickparvar_only"]
proto4 = next(p for p in PROTOCOLS if p["name"] == "four_nickparvar_raw_CLUSTER")
VARIANTS = {"ERM_raw": ("raw", False), "ERM_harm": ("harm", False), "mask_only": ("mask", False), "DRO_only": ("raw", True), "SourceShield": ("mask", True)}

LOCAL_CK = Path("/content/ckpt_local") if not LOCAL_DUMMY else (AO / "ckpt_local"); LOCAL_CK.mkdir(parents=True, exist_ok=True)
def safe_save(obj, dst):
    """Write to local disk first, then copy to Drive with retries (Drive FUSE can reject direct torch.save)."""
    loc = LOCAL_CK / Path(dst).name; torch.save(obj, loc)
    for k in range(4):
        try: _sh.copyfile(loc, str(dst) + ".tmp"); os.replace(str(dst) + ".tmp", dst); return
        except Exception as e: print("drive copy retry", Path(dst).name, type(e).__name__); time.sleep(5 * (k + 1))
    print("WARNING: kept local only:", loc)
def safe_load(dst):
    for cand in (LOCAL_CK / Path(dst).name, Path(dst)):
        if cand.exists():
            try: return torch.load(cand, map_location=DEVICE)
            except Exception:
                try: return torch.load(cand, map_location=DEVICE, weights_only=False)
                except Exception as e: print("load failed", cand, type(e).__name__)
    return None

def train_variant(arch, A, y, g, tr, va, seed, tag, dro):
    set_seed(seed); ck = AO / "checkpoints" / f"{tag}.pt"; part = AO / "checkpoints" / f"{tag}_partial.pt"
    model = timm.create_model(arch, pretrained=not LOCAL_DUMMY, num_classes=4).to(DEVICE)
    s_ = safe_load(ck)
    if s_ is not None: model.load_state_dict(s_["model"]); return model, s_["log"]
    lr = 1e-4 if arch.startswith("convnext") else LR
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=(0.05 if arch.startswith("convnext") else 1e-4))
    steps = max(1, EPOCHS_C * math.ceil(len(tr) / BATCH)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    w = torch.tensor(len(tr) / (4 * np.bincount(y[tr], minlength=4).clip(1)), dtype=torch.float32, device=DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    q = torch.ones(len(ORIGINS), device=DEVICE) / len(ORIGINS); eta = 0.01
    best, best_state, log, bad, ep0 = -1, None, [], 0, 0
    s_ = safe_load(part)
    if s_ is not None:   # resume mid-run (per-epoch checkpoint)
        model.load_state_dict(s_["model"]); opt.load_state_dict(s_["opt"]); sched.load_state_dict(s_["sched"])
        best, best_state, log, bad, ep0, q = s_["best"], s_["best_state"], s_["log"], s_["bad"], s_["epoch"] + 1, s_["q"].to(DEVICE)
    for ep in range(ep0, EPOCHS_C):
        model.train(); perm = np.random.permutation(tr); losses = []
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            x = augment(tens(A[b]).to(DEVICE)); yt = torch.from_numpy(y[b]).long().to(DEVICE); gt = torch.from_numpy(g[b]).long().to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                per = F.cross_entropy(model(x), yt, weight=w, reduction="none").float()
            if dro:
                gl = torch.zeros(len(ORIGINS), device=DEVICE); cnt = torch.bincount(gt, minlength=len(ORIGINS)).float()
                gl.index_add_(0, gt, per); gl = gl / cnt.clamp(min=1)
                q = q * torch.exp(eta * gl.detach() * (cnt > 0)); q = q / q.sum()
                loss = (q * gl).sum()
            else: loss = per.mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.item()))
        vba = float(balanced_accuracy_score(y[va], predict_A(model, A[va]).argmax(1)))
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_ba": vba, "q": [float(v) for v in q.cpu()]})
        if vba > best: best, bad, best_state = vba, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best, "best_state": best_state,
                    "log": log, "bad": bad, "epoch": ep, "q": q.cpu()}, part)
        if bad >= PATIENCE: break
    model.load_state_dict(best_state); safe_save({"model": best_state, "log": log}, ck)
    for pp in (part, LOCAL_CK / part.name):
        try:
            if pp.exists(): pp.unlink()
        except Exception: pass
    return model, log

m4 = task_frame("four", "nickparvar"); y4 = np.full(len(man), -1); y4[m4.img_id.values] = m4.y.values
g4 = np.array([ORIGINS.index(origin(i)) if y4[i] >= 0 else 0 for i in range(len(man))])
X_mask_cache = {}
c2 = []
for arch in ARCHS_C:
    for seed in SEEDS_C:
        _, tr, va, tests = make_split(proto4, seed); te = tests["in_source"]
        for vname, (rep, dro) in VARIANTS.items():
            if rep == "mask":
                need = np.concatenate([tr, va, te])
                if "ids" not in X_mask_cache or not np.isin(need, X_mask_cache["ids"]).all():
                    ids = m4.img_id.values; X_mask_cache = {"ids": ids, "A": X_raw.copy()}
                    X_mask_cache["A"][ids] = masked(X_raw, ids)
                A = X_mask_cache["A"]
            else: A = X_raw if rep == "raw" else X_harm
            if vname == "ERM_raw" and load_full(proto4, arch, seed) is not None and not LOCAL_DUMMY:
                model = load_full(proto4, arch, seed)        # identical setup already trained in FULL
            else:
                model, _ = train_variant(arch, A, y4, g4, tr, va, seed, f"{vname}_{arch}_s{seed}_{SPLIT_SEED}", dro)
            prob = predict_A(model, A[te]); r = metrics(y4[te], prob, 4)
            c2.append({"model": vname, "protocol": "four_cluster", "arch": arch, "seed": seed, "test": "in_source", **r})
            c2 += ext_eval(lambda B_: predict_A(model, B_), rep, "four", {"model": vname, "protocol": "four_cluster", "arch": arch, "seed": seed})
            del model; torch.cuda.empty_cache()
            save_csv(pd.DataFrame(c2), AO / "tables" / "sourceshield_per_seed.csv")
        cstatus(f"C2 SourceShield variants done {arch} s{seed}")
C2 = pd.DataFrame(c2)

# ======================= statistics (L7) =======================
srows = []
for (arch, test), g in C2.groupby(["arch", "test"]):
    base = g[g.model == "ERM_raw"].set_index("seed").ba
    for vname in [v for v in VARIANTS if v != "ERM_raw"]:
        x = g[g.model == vname].set_index("seed").ba; s = base.index.intersection(x.index)
        if len(s) < 2: continue
        d = (x[s] - base[s]).values; bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(2000)]
        srows.append({"arch": arch, "test": test, "variant": vname, "n_seeds": len(s), "mean_diff_ba": d.mean(),
                      "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                      "t_p": _st.ttest_rel(x[s], base[s]).pvalue if d.std() > 0 else np.nan, "d_z": d.mean() / (d.std(ddof=1) + 1e-12)})
S2 = pd.DataFrame(srows)
if len(S2):
    ok = S2.t_p.notna(); S2.loc[ok, "t_p_holm"] = multipletests(S2.loc[ok, "t_p"], method="holm")[1]
save_csv(S2, AO / "tables" / "sourceshield_stats.csv")

# ======================= summary =======================
def mean_tab(df, by):
    return df.groupby(by).agg(ba=("ba", "mean"), sd=("ba", "std"), n=("ba", "size")).reset_index() if len(df) else pd.DataFrame()
lines = ["# ADD-ON C RESULTS — P1 (external sets, SourceShield, foundation model)", f"{time.ctime()} | LOCAL_DUMMY={LOCAL_DUMMY}", "",
         "## External datasets: lineage check", cont.to_markdown(index=False), ""]
ss = S2[(S2.variant == "SourceShield") & S2.test.str.startswith("external")] if len(S2) else pd.DataFrame()
if len(ss):
    hit = ((ss.mean_diff_ba >= 0.05) & (ss.ci_lo > 0)).mean()
    lines += [f"SourceShield vs ERM on external sets: {'🟢' if hit >= 0.5 else '🟡' if (ss.mean_diff_ba > 0).mean() >= 0.5 else '🔴'} "
              f"median ΔBA {ss.mean_diff_ba.median():+.3f}; pre-declared criterion (≥0.05, CI>0) met in {100*hit:.0f}% of arch×dataset", ""]
lines += ["## SourceShield (mean BA)", mean_tab(C2, ["model", "test"]).round(3).to_markdown(index=False), "",
          "## SourceShield statistics", S2.round(4).to_markdown(index=False) if len(S2) else "none", "",
          "## Full models on external sets (mean BA)", mean_tab(C1, ["protocol", "test"]).round(3).to_markdown(index=False), "",
          f"## Frozen foundation model ({fm_name}) (mean BA)", mean_tab(C3, ["protocol", "test"]).round(3).to_markdown(index=False)]
save_text("\n".join(lines), AO / "ADDON_C_RESULTS.md")
zp = ROOT / "P1_addonC_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in AO.rglob("*"):
        if f.is_file() and "checkpoints" not in f.parts: zf.write(f, f.relative_to(ROOT))
cstatus(f"ADD-ON C COMPLETE -> {zp.name}")
