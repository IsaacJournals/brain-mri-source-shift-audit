# %% [markdown]
# # P1 — Base-paper replication: same models, their split vs clean split vs new source
# Re-implements the two classification base papers as closely as their published details allow and evaluates each under three protocols:
# 1. **OFFICIAL**: Nickparvar official train/test folders, validation carved image-level from train (as the base papers did).
#    Also scored on **OFFICIAL-CLEAN**: the subset of official test images with no near-duplicate or same-patient image in training.
# 2. **CLUSTER**: retrained on a split where no duplicate cluster / patient crosses train, validation and test.
# 3. **CROSS**: trained on Nickparvar images NOT derived from Figshare, tested on Figshare (3 tumour classes, 233 patients).
#
# Models (names state what actually runs):
# - **MLHnet-style** re-implementation of Kaprekar et al., Sci Rep 16:24068 (2026): EfficientNet-B0 mid (80-ch) and final (1280-ch) features, CBAM on each, GeM pooling, 4-block transformer on final tokens, fusion of mid + final + CLS. AdamW 1e-4, weight decay 1e-3, batch 32, early stopping (patience 15), their augmentation. Unreported widths are chosen and logged.
# - **Khan-style CNN** re-implementation of Khan et al., Sci Rep 16:19178 (2026): 3 conv layers (3x3, ReLU, 2x2 max-pool), dense 128, dropout 0.5, Adam, 25 epochs; median filter + histogram equalisation + [0,1] scaling. Filter counts were unreported (32/64/128 used, logged).
#
# Reads the shared image cache and manifest written by the main P1 notebook in the same Drive folder; no dataset download and no Kaggle key needed.
# Output: MI10_P1/project_outputs_replication/ + P1_replication_results.zip.

# %%
import os, sys, json, time, math, random, glob, zipfile, subprocess, platform
from pathlib import Path
import numpy as np, pandas as pd
IN_COLAB = ("google.colab" in sys.modules) or os.path.exists("/content/drive/MyDrive") or os.environ.get("P1_IN_COLAB") == "1"   # also true when launched as a subprocess from a Colab cell
LOCAL = os.environ.get("P1_LOCAL_DUMMY") == "1"
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "statsmodels", "tabulate"], check=False)
    if not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive; drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/MI10_P1")
else:
    ROOT = Path(os.environ.get("P1_ROOT", "./MI10_P1_local"))
import torch, torch.nn as nn, torch.nn.functional as F, timm
import scipy.stats as st
from scipy import ndimage as ndi
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from statsmodels.stats.multitest import multipletests

RUN_ACCOUNT = ""   # the Google account you run Colab with (recorded in the run manifest)
SEEDS = [0] if LOCAL else [0, 1, 2]
EPOCHS = {"mlhnet_style": 1 if LOCAL else 30, "khan_cnn": 1 if LOCAL else 25}
PATIENCE = {"mlhnet_style": 15, "khan_cnn": 25}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not LOCAL and DEVICE != "cuda":
    raise RuntimeError("GPU expected. Fix: Runtime -> Change runtime type -> T4 GPU -> Save, then Run all.")
OUT = ROOT / "project_outputs_replication"; (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
print(f"device={DEVICE} torch={torch.__version__} timm={timm.__version__} | seeds={SEEDS} | out={OUT}")
CONFIG_LOG = {"mlhnet_style": {"transformer_dim": 256, "heads": 8, "mlp": 512, "blocks": 4, "fusion_hidden": 256,
                               "pretrained": "ImageNet (timm efficientnet_b0)", "max_epochs": EPOCHS["mlhnet_style"],
                               "note": "paper: 50 epochs max with patience 15; capped at 30 here for compute, logged"},
              "khan_cnn": {"filters": [32, 64, 128], "dense": 128, "dropout": 0.5, "lr": 1e-3, "batch": 32,
                           "epochs": EPOCHS["khan_cnn"], "preprocess": "median 3x3 + histogram equalisation + [0,1]"}}
(OUT / "replication_config.json").write_text(json.dumps(CONFIG_LOG, indent=2))

# %%
# ---------- load shared cache + manifest ----------
caches = sorted((ROOT / "cache").glob("arrays_ALL_*.npz")) or sorted((ROOT / "cache").glob("arrays_*.npz"))
assert caches, "No image cache found. Fix: run the main P1 notebook (STANDARD or FULL) first."
z = np.load(caches[-1]); X = z["X_raw"]
man = pd.read_csv(ROOT / "project_outputs" / "data_audit" / "manifest_with_clusters.csv")
assert len(man) == len(X), f"manifest ({len(man)}) and cache ({len(X)}) differ. Fix: rerun the main notebook so both come from the same run."
print("cache", caches[-1].name, X.shape, "| manifest", man.shape)
CLS4 = ["glioma", "meningioma", "pituitary", "notumor"]; CLS3 = CLS4[:3]

nk = man[(man.source == "nickparvar") & man.label.isin(CLS4)].copy(); nk["y"] = nk.label.map({c: i for i, c in enumerate(CLS4)})
fs = man[(man.source == "figshare") & man.label.isin(CLS3)].copy(); fs["y"] = fs.label.map({c: i for i, c in enumerate(CLS3)})

def assert_disjoint(*idx_sets):
    g = man.set_index("img_id").group
    sets = [set(g.loc[i]) for i in idx_sets]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert not (sets[a] & sets[b]), "LEAKAGE: a duplicate cluster / patient crosses splits"

def make_protocol(name, seed):
    if name == "OFFICIAL":
        tr_all = nk[nk.official_split == "train"]; te = nk[nk.official_split == "test"].img_id.values
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(tr_all, tr_all.y))
        tr, va = tr_all.iloc[a].img_id.values, tr_all.iloc[b].img_id.values
        used = set(man.set_index("img_id").group.loc[np.concatenate([tr, va])])
        clean = np.array([i for i in te if man.group.iloc[i] not in used])
        return dict(train=tr, val=va, tests={"official_test": te, "official_test_clean": clean}, ncls=4)
    if name == "CLUSTER":
        g = nk.group.values
        tr_i, te_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=seed).split(nk, nk.y, g))
        rest = nk.iloc[tr_i]
        a, b = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed + 1).split(rest, rest.y, rest.group))
        tr, va, te = rest.iloc[a].img_id.values, rest.iloc[b].img_id.values, nk.iloc[te_i].img_id.values
        assert_disjoint(tr, va, te)
        return dict(train=tr, val=va, tests={"cluster_test": te}, ncls=4)
    if name == "CROSS":
        fs_groups = set(fs.group)
        pool = nk[nk.label.isin(CLS3) & ~nk.group.isin(fs_groups)].copy(); pool["y"] = pool.label.map({c: i for i, c in enumerate(CLS3)})
        a, b = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(pool, pool.y, pool.group))
        tr, va = pool.iloc[a].img_id.values, pool.iloc[b].img_id.values
        te = fs[~fs.group.isin(set(pool.group))].img_id.values
        assert_disjoint(np.concatenate([tr, va]), te)
        return dict(train=tr, val=va, tests={"cross_figshare": te}, ncls=3, ymap="CLS3")
    raise ValueError(name)

Y = np.full(len(man), -1, np.int64)
def labels_for(p):
    y = np.full(len(man), -1, np.int64)
    if p.get("ymap") == "CLS3":
        lab = man.label.map({c: i for i, c in enumerate(CLS3)})
    else:
        lab = man.label.map({c: i for i, c in enumerate(CLS4)})
    ok = lab.notna(); y[ok.values] = lab[ok].astype(int).values
    return y

# %%
# ---------- Khan-style preprocessing (precomputed once) ----------
def hist_eq(a):
    h = np.bincount(a.ravel(), minlength=256).cumsum(); h = (h - h.min()) * 255 / max(h.max() - h.min(), 1)
    return h.astype(np.uint8)[a]
need = np.unique(np.concatenate([nk.img_id.values, fs.img_id.values]))
XK = np.zeros_like(X)
for i in need:
    XK[i] = hist_eq(ndi.median_filter(X[i], size=3))
print("Khan preprocessing done for", len(need), "images")

# ---------- models ----------
class CBAM(nn.Module):
    def __init__(s, c, r=16):
        super().__init__(); s.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1), nn.ReLU(), nn.Conv2d(c // r, c, 1)); s.sp = nn.Conv2d(2, 1, 7, padding=3)
    def forward(s, x):
        x = x * torch.sigmoid(s.mlp(F.adaptive_avg_pool2d(x, 1)) + s.mlp(F.adaptive_max_pool2d(x, 1)))
        return x * torch.sigmoid(s.sp(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
class GeM(nn.Module):
    def __init__(s, p=3.0): super().__init__(); s.p = nn.Parameter(torch.tensor(p))
    def forward(s, x): return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(s.p), 1).pow(1 / s.p).flatten(1)
class MLHnetStyle(nn.Module):
    def __init__(s, ncls, d=256):
        super().__init__()
        b = timm.create_model("efficientnet_b0", pretrained=not LOCAL)
        s.stem = nn.Sequential(b.conv_stem, b.bn1); s.mid = nn.Sequential(*b.blocks[:4]); s.rest = nn.Sequential(*b.blocks[4:], b.conv_head, b.bn2)
        s.cb_m, s.cb_f = CBAM(80), CBAM(1280); s.gm, s.gf = GeM(), GeM()
        s.proj_m, s.proj_f = nn.Linear(80, d), nn.Linear(1280, d); s.tok = nn.Conv2d(1280, d, 1); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.pos = nn.Parameter(torch.zeros(1, 50, d))
        layer = nn.TransformerEncoderLayer(d, 8, 512, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        s.tr = nn.TransformerEncoder(layer, 4); s.drop = nn.Dropout(0.3)
        s.head = nn.Sequential(nn.Linear(3 * d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, ncls))
    def forward(s, x):
        m = s.mid(s.stem(x)); f = s.rest(m)
        vm = s.proj_m(s.gm(s.cb_m(m))); vf = s.proj_f(s.gf(s.cb_f(f)))
        t = s.tok(f).flatten(2).transpose(1, 2); t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1); t = t + s.pos[:, :t.shape[1]]
        c = s.tr(t)[:, 0]
        return s.head(s.drop(torch.cat([vm, vf, c], 1)))
class KhanCNN(nn.Module):
    def __init__(s, ncls):
        super().__init__()
        L = []; cin = 1
        for c in [32, 64, 128]: L += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]; cin = c
        s.f = nn.Sequential(*L); s.h = nn.Sequential(nn.Flatten(), nn.Linear(128 * 28 * 28, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, ncls))
    def forward(s, x): return s.h(s.f(x))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def batch(arch, idx, train):
    if arch == "khan_cnn":
        x = torch.from_numpy(XK[idx]).float().div(255).unsqueeze(1)
        if train:
            if random.random() < 0.3: x = torch.flip(x, (3,))
        return x
    x = torch.from_numpy(X[idx]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    if train:   # MLHnet augmentation: colour jitter, rotation within +-90, h/v flips p=0.5
        x = x * (1 + (torch.rand(len(x), 1, 1, 1) - 0.5) * 1.2)
        mu = x.mean((1, 2, 3), keepdim=True); x = (x - mu) * (1 + (torch.rand(len(x), 1, 1, 1) - 0.5) * 2.0).clamp(0.1) + mu
        x = x.clamp(0, 1)
        if random.random() < 0.5: x = torch.flip(x, (3,))
        if random.random() < 0.5: x = torch.flip(x, (2,))
        ang = (torch.rand(len(x)) - 0.5) * math.pi
        th = torch.stack([torch.stack([ang.cos(), -ang.sin(), torch.zeros_like(ang)], 1), torch.stack([ang.sin(), ang.cos(), torch.zeros_like(ang)], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    return (x - MEAN) / STD

def predict(model, arch, idx):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE == "cuda"):
        for s in range(0, len(idx), 128):
            out.append(torch.softmax(model(batch(arch, idx[s:s + 128], False).to(DEVICE)).float(), 1).cpu().numpy())
    return np.concatenate(out)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train(arch, p, seed, tag):
    ck = OUT / "checkpoints" / f"{tag}.pt"
    set_seed(seed); ncls = p["ncls"]; y = labels_for(p)
    model = (MLHnetStyle(ncls) if arch == "mlhnet_style" else KhanCNN(ncls)).to(DEVICE)
    if ck.exists():
        try: st_ = torch.load(ck, map_location=DEVICE)
        except Exception: st_ = torch.load(ck, map_location=DEVICE, weights_only=False)   # own checkpoint (L11)
        model.load_state_dict(st_["model"]); return model, st_["log"], y
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) if arch == "mlhnet_style" else torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best, best_state, bad, log, tr = -1, None, 0, [], p["train"]
    for ep in range(EPOCHS[arch]):
        model.train(); perm = np.random.permutation(tr); L = []
        for s in range(0, len(perm), 32):
            b = perm[s:s + 32]
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(model(batch(arch, b, True).to(DEVICE)), torch.from_numpy(y[b]).to(DEVICE))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); L.append(loss.item())
        vba = balanced_accuracy_score(y[p["val"]], predict(model, arch, p["val"]).argmax(1))
        log.append({"epoch": ep, "loss": float(np.mean(L)), "val_ba": float(vba)})
        if vba > best: best, bad, best_state = vba, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= PATIENCE[arch]: break
    model.load_state_dict(best_state); torch.save({"model": best_state, "log": log}, ck)
    return model, log, y

# %%
# ---------- pre-flight (crash check on a tiny subset, 1 epoch; not reported) ----------
_p = make_protocol("OFFICIAL", 0); _p = dict(_p, train=_p["train"][:64], val=_p["val"][:32])
for arch in ["khan_cnn", "mlhnet_style"]:
    _e = dict(EPOCHS); EPOCHS[arch] = 1
    m, _, yy = train(arch, _p, 0, f"preflight_{arch}"); predict(m, arch, _p["tests"]["official_test"][:16]); EPOCHS.update(_e)
    (OUT / "checkpoints" / f"preflight_{arch}.pt").unlink(missing_ok=True)
print("pre-flight OK")

# %%
# ---------- main runs (resumable) ----------
res_f = OUT / "replication_per_seed.csv"
R = pd.read_csv(res_f) if res_f.exists() else pd.DataFrame()
pred_dir = OUT / "predictions"; pred_dir.mkdir(exist_ok=True)
for arch in ["khan_cnn", "mlhnet_style"]:
    for proto in ["OFFICIAL", "CLUSTER", "CROSS"]:
        for seed in SEEDS:
            if len(R) and ((R.arch == arch) & (R.protocol == proto) & (R.seed == seed)).any(): continue
            p = make_protocol(proto, 1000 + seed); t0 = time.time()
            model, log, y = train(arch, p, seed, f"{arch}_{proto}_s{seed}")
            rows = []
            for tname, idx in p["tests"].items():
                if len(idx) == 0: continue
                prob = predict(model, arch, idx); pred = prob.argmax(1)
                try: auc = roc_auc_score(y[idx], prob, multi_class="ovr", average="macro", labels=list(range(p["ncls"])))
                except ValueError: auc = np.nan
                rows.append({"arch": arch, "protocol": proto, "seed": seed, "test": tname, "n": len(idx), "acc": accuracy_score(y[idx], pred),
                             "ba": balanced_accuracy_score(y[idx], pred), "macro_f1": f1_score(y[idx], pred, average="macro"), "auroc": auc,
                             "epochs_run": len(log), "train_min": (time.time() - t0) / 60, "n_train": len(p["train"])})
                pd.DataFrame({"img_id": idx, "y": y[idx], "pred": pred, "arch": arch, "protocol": proto, "seed": seed, "test": tname}).to_parquet(
                    pred_dir / f"{arch}_{proto}_{tname}_s{seed}.parquet")
            R = pd.concat([R, pd.DataFrame(rows)], ignore_index=True); R.to_csv(res_f, index=False)
            print(f"[{arch} {proto} s{seed}] " + ", ".join(f"{r['test']}: acc={r['acc']:.4f} ba={r['ba']:.4f}" for r in rows), flush=True)

# %%
# ---------- aggregate, CIs, tests ----------
g = man.set_index("img_id").group
def boot_ci(arch, proto, test, B=1000 if not LOCAL else 50):
    fs_ = sorted(pred_dir.glob(f"{arch}_{proto}_{test}_s*.parquet"))
    if not fs_: return (np.nan, np.nan)
    d = pd.concat([pd.read_parquet(f) for f in fs_]); d["grp"] = g.loc[d.img_id].values
    by = {k: v for k, v in d.groupby("grp")}; keys = list(by); rng = np.random.default_rng(0); vals = []
    for _ in range(B):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        vals.append(np.mean([accuracy_score(x.y, x.pred) for _, x in s.groupby("seed")]))
    return tuple(np.percentile(vals, [2.5, 97.5]))
A = R.groupby(["arch", "protocol", "test"]).agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"), ba_mean=("ba", "mean"), ba_sd=("ba", "std"),
                                                f1_mean=("macro_f1", "mean"), auroc_mean=("auroc", "mean"), n=("n", "first"), seeds=("seed", "nunique")).reset_index()
A["acc_ci_lo"], A["acc_ci_hi"] = zip(*[boot_ci(r.arch, r.protocol, r.test) for r in A.itertuples()])
A.to_csv(OUT / "replication_aggregate.csv", index=False)

T = []
for arch in R.arch.unique():
    s = R[R.arch == arch]
    off = s[s.test == "official_test"].set_index("seed").acc
    for other in ["official_test_clean", "cluster_test", "cross_figshare"]:
        o = s[s.test == other].set_index("seed").acc; k = off.index.intersection(o.index)
        if len(k) >= 2:
            d = off[k].values - o[k].values
            T.append({"arch": arch, "comparison": f"official_test - {other}", "diff_acc_mean": d.mean(), "diff_sd": d.std(ddof=1),
                      "t_p_one_sided": st.ttest_1samp(d, 0, alternative="greater").pvalue, "n_seeds": len(k), "d_z": d.mean() / (d.std(ddof=1) + 1e-12)})
        elif len(k) == 1:
            T.append({"arch": arch, "comparison": f"official_test - {other}", "diff_acc_mean": float((off[k].values - o[k].values)[0]), "n_seeds": 1})
T = pd.DataFrame(T)
if len(T) and "t_p_one_sided" in T: T["p_holm"] = multipletests(T.t_p_one_sided.fillna(1), method="holm")[1]
T.to_csv(OUT / "replication_tests.csv", index=False)

# %%
# ---------- summary + ZIP ----------
reported = {"mlhnet_style": "Kaprekar 2026: 99.30% mean test accuracy (official 4-class)", "khan_cnn": "Khan 2026: 99.29% (CNN)"}
lines = ["# RESULTS — P1 base-paper replication", f"{time.ctime()} | account {RUN_ACCOUNT} | seeds {SEEDS} | LOCAL_DUMMY={LOCAL}",
         "**NOT FOR PUBLICATION (local dummy run)**" if LOCAL else "Real data; mean ± SD over seeds, 95% cluster-bootstrap CI.", "",
         "| Model | Reported by base paper | Official test acc | Official test, duplicates removed | Clean re-split | Train other sources → Figshare |",
         "|---|---|---|---|---|---|"]
def cell(arch, test):
    r = A[(A.arch == arch) & (A.test == test)]
    if not len(r): return "-"
    r = r.iloc[0]; return f"{100*r.acc_mean:.2f} ± {100*(r.acc_sd if r.acc_sd == r.acc_sd else 0):.2f} [{100*r.acc_ci_lo:.1f}, {100*r.acc_ci_hi:.1f}]"
for arch in ["mlhnet_style", "khan_cnn"]:
    lines.append(f"| {arch} | {reported[arch]} | {cell(arch,'official_test')} | {cell(arch,'official_test_clean')} | {cell(arch,'cluster_test')} | {cell(arch,'cross_figshare')} |")
lines += ["", "## Paired differences (accuracy points, official test minus other)", T.to_markdown(index=False) if len(T) else "n/a",
          "", "Cross-source test is 3-class (glioma/meningioma/pituitary); official and clean are 4-class, so the cross column is not directly comparable in chance level (33% vs 25%)."]
(OUT / "RESULTS_REPLICATION.md").write_text("\n".join(lines)); print("\n".join(lines))
zp = ROOT / "P1_replication_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.rglob("*"):
        if f.is_file() and "checkpoints" not in f.parts: zf.write(f, f.relative_to(ROOT))
print("ZIP", zp, f"{zp.stat().st_size/1e6:.1f} MB")
