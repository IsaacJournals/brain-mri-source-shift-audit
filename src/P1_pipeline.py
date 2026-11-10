# %% [markdown]
# # P1 — Source shortcuts or tumours? A deduplicated, leave-one-source-out benchmark of public brain-tumour MRI classifiers
#
# **Question.** When public brain-tumour MRI classifiers are tested on a *different* public source after cross-corpus deduplication, how much accuracy survives, how much of the label is predictable from non-anatomical cues (background pixels, file format), does canonical harmonisation remove those cues, and how much does patient overlap inflate results when models are retrained?
#
# **Hypotheses (pre-declared; gates in the config cell).**
# - H1: in-source balanced accuracy (BA) − cross-source BA ≥ Δmin (Δmin fixed by the Phase 0 cell, ≤ half the measured gap, default 0.03).
# - H2: background-only / format probes reach BA > 0.60 in-source (binary task) with the 95% CI excluding chance.
# - H3 (two-sided): harmonisation changes cross-source BA and moves probe BA toward chance.
# - H4: image-level Figshare split BA − patient-disjoint split BA > 0.
#
# **Novelty-to-evidence matrix.** C1 (H1) → tables/protocol_results_agg.csv, Table 3. C2 (H2) → tables/probe_results.csv, Table 4. C3 (H3) → same tables, raw vs harmonised. C4 (H4) → tables/patient_vs_image.csv, Table 5. Containment → tables/containment_matrix.csv, Table 2.
#
# **Baselines.** Majority class (naive); logistic regression on intensity histograms; LightGBM on histogram + gradient features (strong classical); ResNet-18, EfficientNet-B0, ConvNeXt-T (ImageNet-pretrained). Identical splits and budgets.
#
# **Threats to validity.** Kaggle sources have no patient IDs (only Figshare does; IDs propagated through duplicate clusters); SARTAJ glioma label noise (reported with and without glioma); 2D slices only; acquisition metadata absent.
#
# > **Reproducibility warning.** SMOKE_TEST output is for plumbing checks only and is NOT FOR PUBLICATION. Only FULL_PAPER_RUN on real data may fill manuscript tables. Run on the Google account confirmed in the config cell.

# %%
# ===================== CONFIG (edit MODE only) =====================
import os as _os
MODE = _os.environ.get("P1_MODE", "FULL_PAPER_RUN")   # "SMOKE_TEST" | "STANDARD_RUN" | "FULL_PAPER_RUN"
RUN_ACCOUNT = ""   # the Google account you run Colab with (recorded in the run manifest)
DATASETS_CONFIRMED = {"date": "2026-11-07", "by": "user", "datasets": [
    "figshare_cheng", "sartaj", "br35h", "nickparvar", "brisc"]}
USE_DRIVE = True               # persist cache/outputs to Google Drive
DRIVE_FOLDER = "/content/drive/MyDrive/MI10_P1"
FORCE_CONTINUE = False         # override futility stop
MODEL_VERSION = "p1-v1.2"      # bump on any code change so stale checkpoints are not reused

PROFILES = {
    "SMOKE_TEST":     dict(seeds=[0], epochs=1, archs=["resnet18"], per_class_cap=40, phase0_epochs=1, phase0_seeds=[0], boot=200),
    "STANDARD_RUN":   dict(seeds=[0, 1], epochs=8, archs=["resnet18"], per_class_cap=None, phase0_epochs=4, phase0_seeds=[0, 1], boot=1000),
    "FULL_PAPER_RUN": dict(seeds=[0, 1, 2, 3, 4], epochs=15, archs=["resnet18", "efficientnet_b0", "convnext_tiny"], per_class_cap=None, phase0_epochs=4, phase0_seeds=[0, 1], boot=2000),
}
P = PROFILES[MODE]
if _os.environ.get("P1_ARCHS"):   # finalise with a subset of architectures (e.g. after a model failed to train)
    P = dict(P, archs=[a.strip() for a in _os.environ["P1_ARCHS"].split(",") if a.strip()])
SPLIT_SEED = {"SMOKE_TEST": 999, "STANDARD_RUN": 1000, "FULL_PAPER_RUN": 2000}[MODE]  # FULL uses fresh in-source partitions

# Pre-declared gates (never edited after FULL starts; changes go to GATE_CHANGELOG)
GATES = dict(
    DELTA_MIN_DEFAULT=0.03,     # minimum meaningful in-vs-cross BA gap
    PROBE_BA_MIN=0.60,          # H2 threshold (binary task)
    ALPHA=0.05,                 # family-wise, Holm
    NCC_THRESHOLD=0.98,         # near-duplicate threshold (sweep 0.90–0.995 reported)
    SUBGROUP_MIN_N=30,
    FLAT_LEARNING_EPS=0.01,
)
GATE_CHANGELOG = []  # (date, change, reason, "pilot-informed")

IMG = 224
DEDUP_SIZE = 64
BATCH = 64
LR = 3e-4
PATIENCE = 4

# %%
# ===================== SETUP =====================
import os, sys, json, time, hashlib, zipfile, shutil, glob, math, random, subprocess, platform, warnings, io
from pathlib import Path
from collections import defaultdict, Counter
warnings.filterwarnings("ignore", category=UserWarning)

IN_COLAB = ("google.colab" in sys.modules) or os.environ.get("P1_IN_COLAB") == "1"
LOCAL_DUMMY = os.environ.get("P1_LOCAL_DUMMY") == "1"   # local plumbing test only; forces SMOKE and blocks publication
if LOCAL_DUMMY:
    MODE = "SMOKE_TEST"; P = PROFILES[MODE]; USE_DRIVE = False

def _pip(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)

if IN_COLAB:
    _pip(["timm", "lightgbm", "h5py", "kagglehub>=0.4.1", "kaggle>=1.8.0", "statsmodels", "tabulate", "pyarrow"])

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import scipy.stats as st
from scipy.io import loadmat
import h5py
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, recall_score
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if MODE != "SMOKE_TEST" and DEVICE != "cuda":
    raise RuntimeError("GPU expected for STANDARD/FULL. Fix: Runtime -> Change runtime type -> T4 GPU, then Run all.")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(0)

print(f"MODE={MODE} | device={DEVICE} | torch={torch.__version__} | timm={timm.__version__} | lightgbm={lgb.__version__}")
print("Manuscript eligibility:", "ELIGIBLE (if all gates PASS)" if MODE == "FULL_PAPER_RUN" and not LOCAL_DUMMY else "NOT FOR PUBLICATION")

# %%
# ===================== DRIVE / DIRECTORIES =====================
if IN_COLAB and USE_DRIVE:
    if not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    ROOT = Path(DRIVE_FOLDER)
else:
    ROOT = Path("./MI10_P1_local")
DATA = Path("/content/p1_data") if IN_COLAB else ROOT / "data"   # raw images on fast local disk; arrays/results go to Drive
CACHE = ROOT / "cache"
OUT = ROOT / "project_outputs"
SUB = {k: OUT / k for k in ["config", "data_audit", "checkpoints", "predictions", "tables", "figures", "models", "logs"]}
for p in [DATA, CACHE, *SUB.values()]:
    p.mkdir(parents=True, exist_ok=True)
lock = ROOT / f"RUN_LOCK_{MODE}.txt"
if lock.exists() and (time.time() - lock.stat().st_mtime) < 600:
    print("WARNING: another session wrote to this folder in the last 10 min. Use one DRIVE_FOLDER per parallel run.")
lock.write_text(time.ctime())
free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
print(f"Outputs: {OUT} | free space {free_gb:.1f} GB | expected need: data ~3 GB, cache ~2.5 GB, outputs ~1 GB")
print({"SMOKE_TEST": "Est. runtime ~10 min", "STANDARD_RUN": "Est. runtime ~1-1.5 h on T4", "FULL_PAPER_RUN": "Est. runtime ~6-8 h on T4 (resume = Run all)"}[MODE])

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"saved {path} shape={df.shape} size={os.path.getsize(path)}B")

def save_text(txt, path):
    Path(path).write_text(txt, encoding="utf-8")
    print(f"saved {path} size={os.path.getsize(path)}B")

# %% [markdown]
# ## Data acquisition
# Sources (all public, de-identified; secondary analysis):
# - Figshare-Cheng brain tumour dataset, v8, DOI 10.6084/m9.figshare.1512427.v8, CC BY 4.0 — direct download.
# - SARTAJ (Kaggle `sartajbhuvaji/brain-tumor-classification-mri`), Br35H (`ahmedhamada0/brain-tumor-detection`), Nickparvar (`masoudnickparvar/brain-tumor-mri-dataset`, DOI 10.34740/KAGGLE/DSV/2645886), BRISC 2025 (`briscdataset/brisc2025`, CC BY 4.0) — Kaggle API. Set Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` in this account (key icon in the left bar). Keys are never printed.

# %%
# ===================== ACQUISITION =====================
KAGGLE = {"sartaj": "sartajbhuvaji/brain-tumor-classification-mri",
          "br35h": "ahmedhamada0/brain-tumor-detection",
          "nickparvar": "masoudnickparvar/brain-tumor-mri-dataset",
          "brisc": "briscdataset/brisc2025"}
FIGSHARE_FILES = {"brainTumorDataPublic_1-766.zip": "https://ndownloader.figshare.com/files/3381290",
                  "brainTumorDataPublic_767-1532.zip": "https://ndownloader.figshare.com/files/3381296",
                  "brainTumorDataPublic_1533-2298.zip": "https://ndownloader.figshare.com/files/3381293",
                  "brainTumorDataPublic_2299-3064.zip": "https://ndownloader.figshare.com/files/3381302"}
PROVENANCE = {}

def kaggle_auth():
    """Accepts the new Kaggle API token (secret KAGGLE_API_TOKEN) or legacy KAGGLE_USERNAME + KAGGLE_KEY."""
    if "google.colab" in sys.modules:
        from google.colab import userdata
        for name in ["KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"]:
            try:
                v = userdata.get(name)
                if v: os.environ[name] = v.strip()
            except Exception:
                pass
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("Kaggle auth: API token (value not printed)"); return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("Kaggle auth: legacy username/key (values not printed)"); return
    raise RuntimeError("Kaggle credentials missing. Fix: in Colab Secrets (key icon) add KAGGLE_API_TOKEN "
                       "(Kaggle > Settings > API Tokens > Generate) with notebook access ON, then Run all.")

def fetch_kaggle(name, slug):
    dest = DATA / name
    done = dest / ".complete"
    if done.exists():
        return dest
    import kagglehub
    path = Path(kagglehub.dataset_download(slug))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    meta = {"slug": slug, "kagglehub_path": str(path), "version_hint": path.name, "downloaded": time.ctime()}
    try:
        r = subprocess.run(["kaggle", "datasets", "metadata", slug, "-p", str(dest)], capture_output=True, text=True, timeout=120)
        mfile = dest / "dataset-metadata.json"
        if mfile.exists():
            m = json.loads(mfile.read_text())
            meta["licenses"] = m.get("licenses"); meta["title"] = m.get("title")
    except Exception as e:
        meta["licenses"] = f"metadata lookup failed: {type(e).__name__}"
    (dest / "provenance.json").write_text(json.dumps(meta, indent=2))
    done.write_text("ok")
    return dest

def http_get(url, out, retries=4):
    import urllib.request
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-notebook"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
                ctype = r.headers.get("Content-Type", "")
                first = r.read(1 << 16)
                if first[:15].lower().startswith(b"<!doctype html") or first[:6].lower().startswith(b"<html"):
                    raise RuntimeError(f"HTML returned instead of data (content-type {ctype})")
                f.write(first); shutil.copyfileobj(r, f)
            return out
        except Exception as e:
            print(f"  retry {k+1}/{retries} for {url}: {e}")
            time.sleep(2 ** k)
    raise RuntimeError(f"Download failed: {url}. Fix: check network, then Run all (finished files are skipped).")

def fetch_figshare():
    dest = DATA / "figshare_cheng"
    done = dest / ".complete"
    if done.exists():
        return dest
    raw = dest / "raw"; raw.mkdir(parents=True, exist_ok=True)
    for fn, url in FIGSHARE_FILES.items():
        zp = raw / fn
        if not zp.exists() or zp.stat().st_size < 1e8:
            print("downloading", fn); http_get(url, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(raw / fn.replace(".zip", ""))
    # .mat (v7.3, HDF5) -> PNG + metadata
    pngdir = dest / "png"; pngdir.mkdir(exist_ok=True)
    rows = []
    lab = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    for mp in sorted(raw.rglob("*.mat")):
        try:
            with h5py.File(mp, "r") as f:
                cj = f["cjdata"]
                label = int(np.array(cj["label"]).squeeze())
                pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).squeeze())
                img = np.array(cj["image"]).T.astype(np.float32)   # MATLAB column-major -> transpose
        except OSError:
            m = loadmat(mp)["cjdata"][0, 0]
            label = int(m["label"].squeeze()); pid = str(m["PID"].squeeze()); img = m["image"].astype(np.float32)
        lo, hi = np.percentile(img, 0.0), np.percentile(img, 100.0)
        u8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        outp = pngdir / lab[label] / f"{mp.stem}.png"; outp.parent.mkdir(exist_ok=True)
        Image.fromarray(u8).save(outp)
        rows.append({"file": str(outp), "label": lab[label], "pid": pid, "mat": mp.name})
    pd.DataFrame(rows).to_csv(dest / "figshare_index.csv", index=False)
    (dest / "provenance.json").write_text(json.dumps({"doi": "10.6084/m9.figshare.1512427.v8", "license": "CC BY 4.0",
                                                      "n_mat": len(rows), "downloaded": time.ctime()}, indent=2))
    done.write_text("ok")
    return dest

def make_dummy_sources():
    """LOCAL plumbing test only (P1_LOCAL_DUMMY=1). Synthetic images; never reaches publication."""
    rng = np.random.default_rng(0)
    def img(kind, fmt):
        a = (rng.random((180, 200)) * 40).astype(np.uint8)
        yy, xx = np.mgrid[:180, :200]
        ry, rx, oy, ox = rng.integers(45, 80), rng.integers(50, 90), rng.integers(70, 110), rng.integers(80, 120)
        a[((yy - oy) ** 2 / ry ** 2 + (xx - ox) ** 2 / rx ** 2) < 1] += 60
        a = (a.astype(np.float32) + rng.normal(0, 25, a.shape) + np.sin(xx / rng.uniform(3, 15)) * 30).clip(0, 255).astype(np.uint8)
        if kind != "notumor":
            cy, cx = rng.integers(60, 120), rng.integers(60, 140)
            a[((yy - cy) ** 2 + (xx - cx) ** 2) < rng.integers(80, 300)] = 230
        return Image.fromarray(a)
    layout = {"sartaj": [("Training/glioma_tumor", "glioma"), ("Training/meningioma_tumor", "meningioma"), ("Training/no_tumor", "notumor"),
                         ("Training/pituitary_tumor", "pituitary"), ("Testing/glioma_tumor", "glioma"), ("Testing/no_tumor", "notumor")],
              "br35h": [("yes", "glioma"), ("no", "notumor")],
              "nickparvar": [("Training/glioma", "glioma"), ("Training/meningioma", "meningioma"), ("Training/notumor", "notumor"),
                             ("Training/pituitary", "pituitary"), ("Testing/glioma", "glioma"), ("Testing/notumor", "notumor"),
                             ("Testing/meningioma", "meningioma"), ("Testing/pituitary", "pituitary")],
              "brisc": [("classification_task/train/glioma", "glioma"), ("classification_task/test/no_tumor", "notumor")]}
    for src, folders in layout.items():
        for sub, kind in folders:
            d = DATA / src / sub; d.mkdir(parents=True, exist_ok=True)
            for i in range(12):
                img(kind, "jpg").save(d / f"img_{i}.jpg", quality=int(rng.integers(70, 95)))
        (DATA / src / ".complete").write_text("dummy")
    # copy a few nickparvar images into sartaj to create cross-corpus duplicates
    for i in range(4):
        shutil.copy(DATA / "nickparvar/Training/glioma" / f"img_{i}.jpg", DATA / "sartaj/Training/glioma_tumor" / f"dup_{i}.jpg")
    fs = DATA / "figshare_cheng" / "png"; rows = []
    for lab_ in ["glioma", "meningioma", "pituitary"]:
        (fs / lab_).mkdir(parents=True, exist_ok=True)
        for i in range(15):
            pth = fs / lab_ / f"{lab_}_{i}.png"; img(lab_, "png").save(pth)
            rows.append({"file": str(pth), "label": lab_, "pid": f"P{i % 6}{lab_[0]}", "mat": f"{i}.mat"})
    pd.DataFrame(rows).to_csv(DATA / "figshare_cheng" / "figshare_index.csv", index=False)
    (DATA / "figshare_cheng" / ".complete").write_text("dummy")

if LOCAL_DUMMY:
    make_dummy_sources()
else:
    kaggle_auth()
    for name, slug in KAGGLE.items():
        print("fetching", name); fetch_kaggle(name, slug)
    print("fetching figshare_cheng"); fetch_figshare()
print("acquisition done")

# %%
# ===================== MANIFEST + PROVENANCE =====================
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def infer_label(parts):
    s = [p.lower() for p in parts]
    j = " ".join(s)
    if any(k in j for k in ["mask", "segmentation", "label_", "/pred"]) or "pred" in s:
        return None
    if "glioma" in j: return "glioma"
    if "meningioma" in j: return "meningioma"
    if "pituitary" in j: return "pituitary"
    if any(k in j for k in ["no_tumor", "notumor", "no tumor", "no_tumour"]) or "no" in s: return "notumor"
    if "yes" in s: return "tumor_unspec"
    return None

def infer_split(parts):
    s = [p.lower() for p in parts]
    if any(p in ("training", "train") for p in s): return "train"
    if any(p in ("testing", "test") for p in s): return "test"
    return "none"

def file_meta(path):
    d = {"filesize": os.path.getsize(path), "ext": Path(path).suffix.lower()}
    try:
        with Image.open(path) as im:
            d.update(width=im.width, height=im.height, pil_mode=im.mode, fmt=im.format or "")
            q = getattr(im, "quantization", None)
            d["jpeg_q_mean"] = float(np.mean(q[0])) if q else -1.0
    except Exception as e:
        d.update(width=-1, height=-1, pil_mode="ERR", fmt="ERR", jpeg_q_mean=-1.0, error=str(e))
    d["bytes_per_px"] = d["filesize"] / max(d.get("width", 1) * d.get("height", 1), 1)
    return d

rows, skipped = [], Counter()
for src in ["sartaj", "br35h", "nickparvar", "brisc"]:
    for fp in sorted((DATA / src).rglob("*")):
        if fp.suffix.lower() not in IMG_EXT: continue
        rel = fp.relative_to(DATA / src).parts
        lab = infer_label(rel)
        if lab is None: skipped[(src, "/".join(rel[:-1]))] += 1; continue
        rows.append({"source": src, "path": str(fp), "label": lab, "official_split": infer_split(rel), "pid": ""})
fsi = pd.read_csv(DATA / "figshare_cheng" / "figshare_index.csv")
for r in fsi.itertuples():
    rows.append({"source": "figshare", "path": r.file, "label": r.label, "official_split": "none", "pid": str(r.pid)})
man = pd.DataFrame(rows)
if man.empty:
    raise RuntimeError("No images found. Fix: check the download step output and folder layout.")
if P["per_class_cap"]:
    keep = []
    for _, g in man.groupby(["source", "label"]):
        keep += list(g.sample(min(len(g), P["per_class_cap"]), random_state=0).index)
    man = man.loc[sorted(keep)].reset_index(drop=True)
meta = pd.DataFrame([file_meta(p) for p in man.path])
man = pd.concat([man.reset_index(drop=True), meta], axis=1)
bad = man[man.pil_mode == "ERR"]
if len(bad): print(f"WARNING: {len(bad)} unreadable files excluded"); man = man[man.pil_mode != "ERR"].reset_index(drop=True)
man["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in man.path]
man["img_id"] = np.arange(len(man))
counts = man.groupby(["source", "official_split", "label"]).size().reset_index(name="n")
save_csv(counts, SUB["tables"] / "corpus_counts_raw.csv")
save_csv(pd.DataFrame([{"source": s, "folder": f, "n_skipped": n} for (s, f), n in skipped.items()]), SUB["data_audit"] / "skipped_folders.csv")

prov_lines = ["# DATA_PROVENANCE", f"Run: {time.ctime()} | mode {MODE} | account {RUN_ACCOUNT} | LOCAL_DUMMY={LOCAL_DUMMY}", ""]
for src in ["figshare_cheng", "sartaj", "br35h", "nickparvar", "brisc"]:
    pj = DATA / src / "provenance.json"
    prov_lines.append(f"## {src}\n{pj.read_text() if pj.exists() else 'dummy/local' }\n")
prov_lines.append("## Counts\n" + counts.to_string(index=False))
save_text("\n".join(prov_lines), SUB["data_audit"] / "DATA_PROVENANCE.md")
print(counts.pivot_table(index=["source", "official_split"], columns="label", values="n", fill_value=0))

# %%
# ===================== IMAGE CACHE (raw + harmonised) =====================
def otsu(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum(); w = np.cumsum(p); mu = np.cumsum(p * np.arange(256)); mt = mu[-1]
    sb = (mt * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return int(np.argmax(sb))

from scipy import ndimage as _ndi
def brain_mask(a):
    b = _ndi.gaussian_filter(a.astype(np.float32), 2).astype(np.uint8)
    m = b > otsu(b)
    return _ndi.maximum_filter(m, size=15)   # separable dilation, same 15x15 footprint as before

def load_gray(path):
    with Image.open(path) as im:
        return np.array(im.convert("L"))

def raw_repr(a):
    return np.array(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR))

def harm_repr(a):
    m = brain_mask(a)
    ys, xs = np.where(m)
    if len(ys) > 50:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        py, px = int(0.05 * (y1 - y0)), int(0.05 * (x1 - x0))
        a = a[max(0, y0 - py): y1 + py + 1, max(0, x0 - px): x1 + px + 1]
    a = np.array(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR)).astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def dedup_vec(a):
    v = np.array(Image.fromarray(a).resize((DEDUP_SIZE, DEDUP_SIZE), Image.BILINEAR)).astype(np.float32).ravel()
    v -= v.mean(); n = np.linalg.norm(v)
    return v / n if n > 0 else v

def bg_features(a):
    a = np.array(Image.fromarray(a).resize((256, 256), Image.BILINEAR))
    m = brain_mask(a); bg = a[~m]
    h = np.histogram(bg, bins=32, range=(0, 256))[0].astype(np.float32); h /= max(h.sum(), 1)
    e = 8
    border = [a[:e].mean(), a[-e:].mean(), a[:, :e].mean(), a[:, -e:].mean()]
    return np.concatenate([h, [bg.mean() if bg.size else 0, bg.std() if bg.size else 0, (~m).mean()], border]).astype(np.float32)

def anat_features(a224):
    h = np.histogram(a224, bins=64, range=(0, 256))[0].astype(np.float32); h /= h.sum()
    gy, gx = np.gradient(a224.astype(np.float32)); g = np.hypot(gx, gy)
    hg = np.histogram(g, bins=16, range=(0, 128))[0].astype(np.float32); hg /= hg.sum()
    return np.concatenate([h, hg])

cache_key = hashlib.md5((MODEL_VERSION + str(len(man)) + "".join(man.md5.values[:50])).encode()).hexdigest()[:10]
cfile = CACHE / (f"arrays_ALL_{cache_key}.npz" if not P["per_class_cap"] else f"arrays_{MODE}_{cache_key}.npz")
if cfile.exists():
    z = np.load(cfile); X_raw, X_harm, V, BG_raw, BG_harm, A_raw, A_harm = (z[k] for k in ["X_raw", "X_harm", "V", "BG_raw", "BG_harm", "A_raw", "A_harm"])
    print("loaded cache", cfile)
else:
    N = len(man)
    from concurrent.futures import ProcessPoolExecutor
    def _one(pth):
        a = load_gray(pth); h = harm_repr(a)
        return raw_repr(a), h, dedup_vec(a).astype(np.float16), bg_features(a), bg_features(h), anat_features(raw_repr(a)), anat_features(h)
    X_raw = np.zeros((N, IMG, IMG), np.uint8); X_harm = np.zeros_like(X_raw)
    V = np.zeros((N, DEDUP_SIZE * DEDUP_SIZE), np.float16); BG_raw = np.zeros((N, 39), np.float32); BG_harm = np.zeros_like(BG_raw)
    A_raw = np.zeros((N, 80), np.float32); A_harm = np.zeros_like(A_raw)
    t0 = time.time(); workers = max(1, os.cpu_count() or 1)
    print(f"  preprocessing {N} images on {workers} CPU workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(_one, list(man.path), chunksize=64)):
            X_raw[i], X_harm[i], V[i], BG_raw[i], BG_harm[i], A_raw[i], A_harm[i] = out
            if i % 2000 == 0: print(f"  cached {i}/{N} | {time.time()-t0:.0f}s", flush=True)
    np.savez(cfile, X_raw=X_raw, X_harm=X_harm, V=V, BG_raw=BG_raw, BG_harm=BG_harm, A_raw=A_raw, A_harm=A_harm)
    print("saved cache", cfile, f"{os.path.getsize(cfile)/1e9:.2f} GB")

# %%
# ===================== CROSS-CORPUS NEAR-DUPLICATE CLUSTERS =====================
class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b):
        a, b = s.f(a), s.f(b)
        if a != b: s.p[b] = a

def dihedral(v):
    m = v.reshape(-1, DEDUP_SIZE, DEDUP_SIZE)
    outs = []
    for k in range(4):
        r = torch.rot90(m, k, dims=(1, 2)); outs += [r, torch.flip(r, dims=(2,))]
    return [o.reshape(len(v), -1) for o in outs]

def near_pairs(V, thr, chunk=2048):
    dev = DEVICE; dt = torch.float16 if dev == "cuda" else torch.float32
    Vt = torch.tensor(V, dtype=dt, device=dev)
    best = []  # (i, j, sim) with i<j
    sims_max = torch.full((len(V),), -1.0, device=dev)  # max sim to any other image, for the threshold sweep
    for s in range(0, len(V), chunk):
        q = Vt[s:s + chunk]
        smax = None
        for qt in dihedral(q):
            sim = qt @ Vt.T
            smax = sim if smax is None else torch.maximum(smax, sim)
        idx = torch.arange(s, s + len(q), device=dev)
        smax[torch.arange(len(q)), idx] = -1
        sims_max[s:s + len(q)] = smax.max(1).values.float()
        ii, jj = torch.where(smax >= thr - 0.10)
        vals = smax[ii, jj].float()
        for a, b, v in zip((ii + s).tolist(), jj.tolist(), vals.tolist()):
            if a < b: best.append((a, b, v))
    return best, sims_max.cpu().numpy()

t0 = time.time()
pairs, maxsim = near_pairs(V, GATES["NCC_THRESHOLD"])
print(f"candidate pairs {len(pairs)} in {time.time()-t0:.0f}s")
pairs_df = pd.DataFrame(pairs, columns=["i", "j", "ncc"])
md5_groups = man.groupby("md5").img_id.apply(list)

def clusters_at(thr):
    uf = UF(len(man))
    for g in md5_groups:
        for x in g[1:]: uf.u(g[0], x)
    for r in pairs_df[pairs_df.ncc >= thr].itertuples(): uf.u(r.i, r.j)
    return np.array([uf.f(i) for i in range(len(man))]), uf

sweep = []
for thr in [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
    cl, _ = clusters_at(thr)
    sweep.append({"threshold": thr, "n_clusters": len(set(cl)), "images_in_multi_clusters": int((pd.Series(cl).map(pd.Series(cl).value_counts()) > 1).sum())})
save_csv(pd.DataFrame(sweep), SUB["tables"] / "ncc_threshold_sweep.csv")

dup_cluster, uf = clusters_at(GATES["NCC_THRESHOLD"])
# merge figshare patients into groups: same PID -> same group; PIDs propagate to all cluster members
for pid, g in man[man.pid != ""].groupby("pid").img_id:
    g = list(g)
    for x in g[1:]: uf.u(g[0], x)
group = np.array([uf.f(i) for i in range(len(man))])
man["dup_cluster"] = dup_cluster; man["group"] = group
pid_by_group = man[man.pid != ""].groupby("group").pid.first()
man["pid_propagated"] = man.group.map(pid_by_group).fillna("")
man["exact_dup_count"] = man.md5.map(man.md5.value_counts()) - 1

srcs = sorted(man.source.unique())
cont = []
for a in srcs:
    ga = set(man.loc[man.source == a, "dup_cluster"])
    for b in srcs:
        if a == b: continue
        mb = man[man.source == b]
        cont.append({"container_A": a, "corpus_B": b, "pct_B_in_A": round(100 * mb.dup_cluster.isin(ga).mean(), 2), "n_B": len(mb),
                     "pct_B_exact_in_A": round(100 * mb.md5.isin(set(man.loc[man.source == a, "md5"])).mean(), 2)})
cont = pd.DataFrame(cont); save_csv(cont, SUB["tables"] / "containment_matrix.csv")
within = []
for s in ["nickparvar", "brisc", "sartaj"]:
    m = man[man.source == s]
    tr = set(m.loc[m.official_split == "train", "dup_cluster"]); te = m[m.official_split == "test"]
    if len(te): within.append({"corpus": s, "n_test": len(te), "pct_test_with_train_twin": round(100 * te.dup_cluster.isin(tr).mean(), 2),
                               "pct_test_exact_in_train": round(100 * te.md5.isin(set(m.loc[m.official_split == "train", "md5"])).mean(), 2)})
save_csv(pd.DataFrame(within), SUB["tables"] / "official_split_leakage.csv")
save_csv(man.drop(columns=[]), SUB["data_audit"] / "manifest_with_clusters.csv")
dedup_counts = man.groupby("source").agg(n_images=("img_id", "size"), n_dup_clusters=("dup_cluster", "nunique"), n_groups=("group", "nunique")).reset_index()
save_csv(dedup_counts, SUB["tables"] / "corpus_counts.csv")
print(cont.pivot(index="container_A", columns="corpus_B", values="pct_B_in_A"))

# %%
# ===================== SPLITS (group-aware) =====================
TUMOUR = {"glioma", "meningioma", "pituitary", "tumor_unspec"}
TYPES = ["glioma", "meningioma", "pituitary"]

def task_frame(task, source):
    m = man[man.source == source].copy()
    if task == "bin":
        m["y"] = m.label.isin(TUMOUR).astype(int)
    elif task == "type":
        m = m[m.label.isin(TYPES)].copy(); m["y"] = m.label.map({k: i for i, k in enumerate(TYPES)})
    elif task == "four":
        m = m[m.label.isin(TYPES + ["notumor"])].copy(); m["y"] = m.label.map({k: i for i, k in enumerate(TYPES + ["notumor"])})
    return m

def group_split(m, seed, test_frac=0.15, val_frac=0.15, groups_col="group"):
    """train/val/test with no group shared; stratified on y."""
    n_groups = m[groups_col].nunique()
    if n_groups < 6:
        raise RuntimeError(f"Only {n_groups} independent groups in this source/task; a grouped split is not possible. "
                           "Check the NCC threshold sweep (images may be over-merged) and the label mapping.")
    n_te = min(max(2, round(1 / test_frac)), n_groups // 2)
    sgk = StratifiedGroupKFold(n_splits=n_te, shuffle=True, random_state=seed)
    tr_idx, te_idx = next(sgk.split(m, m.y, m[groups_col]))
    rest = m.iloc[tr_idx]
    n_va = min(max(2, round((1 - test_frac) / val_frac)), rest[groups_col].nunique() // 2)
    sgk2 = StratifiedGroupKFold(n_splits=n_va, shuffle=True, random_state=seed + 1)
    a, b = next(sgk2.split(rest, rest.y, rest[groups_col]))
    return rest.iloc[a].img_id.values, rest.iloc[b].img_id.values, m.iloc[te_idx].img_id.values

def image_split(m, seed):
    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=seed)
    a, te = next(sss.split(m, m.y)); rest = m.iloc[a]
    sss2 = StratifiedShuffleSplit(1, test_size=0.15 / 0.85, random_state=seed + 1)
    b, c = next(sss2.split(rest, rest.y))
    return rest.iloc[b].img_id.values, rest.iloc[c].img_id.values, m.iloc[te].img_id.values

def assert_disjoint(tr, va, te, col="group"):
    g = man.set_index("img_id")[col]
    s_tr, s_va, s_te = set(g.loc[tr]), set(g.loc[va]), set(g.loc[te])
    assert not (s_tr & s_te) and not (s_va & s_te) and not (s_tr & s_va), f"LEAKAGE: {col} shared across splits"

# Protocol registry: each entry trains once and is evaluated on its in-source test plus cross-source tests
PROTOCOLS = []
for rep in ["raw", "harm"]:
    for A, B in [("br35h", "sartaj"), ("sartaj", "br35h")]:
        PROTOCOLS.append(dict(name=f"bin_{A}_{rep}", task="bin", train_src=A, cross=[B], rep=rep, split="group"))
    for A, B in [("figshare", "sartaj"), ("sartaj", "figshare")]:
        PROTOCOLS.append(dict(name=f"type_{A}_{rep}", task="type", train_src=A, cross=[B], rep=rep, split="group"))
PROTOCOLS.append(dict(name="type_figshare_raw_IMAGELEVEL", task="type", train_src="figshare", cross=[], rep="raw", split="image"))
PROTOCOLS.append(dict(name="four_nickparvar_raw_OFFICIAL", task="four", train_src="nickparvar", cross=[], rep="raw", split="official"))
PROTOCOLS.append(dict(name="four_nickparvar_raw_CLUSTER", task="four", train_src="nickparvar", cross=[], rep="raw", split="group"))

def make_split(proto, seed):
    m = task_frame(proto["task"], proto["train_src"])
    if proto["split"] == "group":
        tr, va, te = group_split(m, SPLIT_SEED + seed); assert_disjoint(tr, va, te)
    elif proto["split"] == "image":
        tr, va, te = image_split(m, SPLIT_SEED + seed)
    else:  # official folders; validation carved image-level from train (as the base papers do)
        trm, te = m[m.official_split == "train"], m[m.official_split == "test"].img_id.values
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=SPLIT_SEED + seed)
        a, b = next(sss.split(trm, trm.y)); tr, va = trm.iloc[a].img_id.values, trm.iloc[b].img_id.values
    tests = {"in_source": te}
    used_groups = set(man.set_index("img_id").group.loc[np.concatenate([tr, va])])
    for B in proto["cross"]:
        mb = task_frame(proto["task"], B)
        tests[f"cross_{B}"] = mb[~mb.group.isin(used_groups)].img_id.values
        tests[f"cross_{B}_contained_removed_n"] = int(mb.group.isin(used_groups).sum())
    if proto["split"] == "official":  # clean subset of the official test: no group shared with train/val
        tests["official_test_clean"] = np.array([i for i in te if man.group.iloc[i] not in used_groups])
    return m, tr, va, tests

split_rows = []
for pr in PROTOCOLS:
    _, tr, va, tests = make_split(pr, 0)
    split_rows.append({"protocol": pr["name"], "n_train": len(tr), "n_val": len(va),
                       **{k: (len(v) if hasattr(v, "__len__") else v) for k, v in tests.items()}})
save_csv(pd.DataFrame(split_rows), SUB["tables"] / "split_table.csv")

# %%
# ===================== METRICS / MODELS =====================
def metrics(y, prob, n_classes):
    pred = prob.argmax(1)
    out = {"ba": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "n": len(y)}
    try:
        out["auroc"] = roc_auc_score(y, prob[:, 1]) if n_classes == 2 else roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    except ValueError:
        out["auroc"] = np.nan
    conf = prob.max(1); acc = (pred == y).astype(float); bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any(): ece += k.mean() * abs(acc[k].mean() - conf[k].mean())
    out["ece"] = ece
    rec = recall_score(y, pred, average=None, labels=list(range(n_classes)), zero_division=0)
    for c, r in enumerate(rec): out[f"recall_c{c}"] = r
    return out

NCLS = {"bin": 2, "type": 3, "four": 4}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def to_tensor(Xu8, idx):
    x = torch.from_numpy(Xu8[idx]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD

def augment(x):
    if random.random() < 0.5: x = torch.flip(x, dims=(3,))
    ang = (torch.rand(len(x), device=x.device) - 0.5) * (2 * math.pi * 10 / 360)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1), torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")

def predict(model, Xu8, idx):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        for s in range(0, len(idx), 128):
            out.append(torch.softmax(model(to_tensor(Xu8, idx[s:s + 128]).to(DEVICE)).float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))

def train_cnn(arch, Xu8, y_all, tr, va, ncls, seed, epochs, tag):
    set_seed(seed)
    ck = SUB["checkpoints"] / f"{tag}.pt"
    model = timm.create_model(arch, pretrained=not LOCAL_DUMMY, num_classes=ncls).to(DEVICE)
    if ck.exists():
        # checkpoints are written by this pipeline into the user's own Drive; older ones hold numpy scalars in the log
        try: state = torch.load(ck, map_location=DEVICE)
        except Exception: state = torch.load(ck, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        return model, [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in state["log"]]
    opt = torch.optim.AdamW(model.parameters(), lr=(1e-4 if arch.startswith('convnext') else LR), weight_decay=(0.05 if arch.startswith('convnext') else 1e-4))
    steps = max(1, epochs * math.ceil(len(tr) / BATCH)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    w = torch.tensor(len(tr) / (ncls * np.bincount(y_all[tr], minlength=ncls).clip(1)), dtype=torch.float32, device=DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    best, best_state, log, bad = -1, None, [], 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr); losses = []
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            x = augment(to_tensor(Xu8, b).to(DEVICE)); yt = torch.from_numpy(y_all[b]).long().to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                loss = F.cross_entropy(model(x), yt, weight=w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        pv = predict(model, Xu8, va); vba = balanced_accuracy_score(y_all[va], pv.argmax(1))
        ptr = predict(model, Xu8, tr[:min(len(tr), 1000)]); tba = balanced_accuracy_score(y_all[tr[:min(len(tr), 1000)]], ptr.argmax(1))
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "train_ba": float(tba), "val_ba": float(vba), "sec": time.time() - t0})
        if vba > best: best, bad = vba, 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "log": log}, ck)
    return model, log

def learning_flag(log):
    if len(log) < 2: return "N/A at SMOKE resolution"
    v = [r["val_ba"] for r in log]; gap = log[-1]["train_ba"] - log[-1]["val_ba"]
    # flat = no improvement AND low validation performance (a pretrained model can peak at epoch 0)
    if max(v) - v[0] < GATES["FLAT_LEARNING_EPS"] and max(v) < 0.70: return "FLAT-LEARNING"
    if gap > 0.15: return "OVERFIT"
    return "OK"

# %%
# ===================== PHASE 0 FEASIBILITY =====================
def run_protocol(pr, seed, arch, epochs, prefix=""):
    m, tr, va, tests = make_split(pr, seed)
    Xu8 = X_raw if pr["rep"] == "raw" else X_harm
    y_all = np.full(len(man), -1, dtype=np.int64); y_all[m.img_id.values] = m.y.values
    for k, v in tests.items():
        if k.startswith("cross_") and isinstance(v, np.ndarray):
            mb = task_frame(pr["task"], k.replace("cross_", "")); y_all[mb.img_id.values] = mb.y.values
    ver = "p1-v1.3cnx" if arch.startswith("convnext") else MODEL_VERSION   # ConvNeXt retrained with fixed LR; others reuse v1.2 checkpoints
    tag = f"{prefix}{pr['name']}_{arch}_s{seed}_{MODE}_{ver}_{SPLIT_SEED}"
    t0 = time.time()
    model, log = train_cnn(arch, Xu8, y_all, tr, va, NCLS[pr["task"]], seed, epochs, tag)
    train_sec = time.time() - t0
    res, preds = [], []
    for tname, idx in tests.items():
        if not isinstance(idx, np.ndarray) or len(idx) == 0: continue
        t1 = time.time(); prob = predict(model, Xu8, idx); inf_ms = 1000 * (time.time() - t1) / len(idx)
        r = metrics(y_all[idx], prob, NCLS[pr["task"]])
        if pr["task"] == "type":  # report without glioma (SARTAJ glioma label noise)
            k = y_all[idx] != 0
            r["ba_excl_glioma"] = recall_score(y_all[idx][k], prob[k].argmax(1), labels=[1, 2], average="macro", zero_division=0) if k.any() else np.nan
        res.append({"protocol": pr["name"], "task": pr["task"], "train_src": pr["train_src"], "rep": pr["rep"], "split": pr["split"],
                    "arch": arch, "seed": seed, "test": tname, "ver": ver, "train_sec": train_sec, "inf_ms_per_img": inf_ms,
                    "n_params": sum(p.numel() for p in model.parameters()), "learning": learning_flag(log), **r})
        preds.append(pd.DataFrame({"img_id": idx, "y": y_all[idx], "pred": prob.argmax(1), "pmax": prob.max(1),
                                   **{f"p{c}": prob[:, c] for c in range(prob.shape[1])},
                                   "protocol": pr["name"], "arch": arch, "seed": seed, "test": tname}))
    for r in log: r.update(protocol=pr["name"], arch=arch, seed=seed)
    return res, pd.concat(preds) if preds else pd.DataFrame(), log

def probe_eval(feats, task, src, cross_src, seed=0):
    """LightGBM probe, grouped CV in-source (out-of-fold) + train-all/test-cross."""
    m = task_frame(task, src); X = feats[m.img_id.values]; y = m.y.values; g = m.group.values
    oof = np.zeros((len(m), NCLS[task]))
    k = min(5, max(2, len(set(g)) // 5))
    for a, b in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, g):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed).fit(X[a], y[a])
        oof[b] = clf.predict_proba(X[b])
    out = [{"test": "in_source_groupedCV", **metrics(y, oof, NCLS[task])}]
    if cross_src:
        mb = task_frame(task, cross_src); mb = mb[~mb.group.isin(set(g))]
        if len(mb) and len(np.unique(y)) > 1:
            clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed).fit(X, y)
            out.append({"test": f"cross_{cross_src}", **metrics(mb.y.values, clf.predict_proba(feats[mb.img_id.values]), NCLS[task])})
    return out, (m.img_id.values, y, oof)

FMT_COLS = ["width", "height", "filesize", "bytes_per_px", "jpeg_q_mean"]
FMT = np.zeros((len(man), len(FMT_COLS) + 2), np.float32)
FMT[:, :len(FMT_COLS)] = man[FMT_COLS].values
FMT[:, -2] = man.pil_mode.astype("category").cat.codes.values; FMT[:, -1] = man.ext.astype("category").cat.codes.values

# ---- approved add-on (R5, R7): runs on the saved FULL checkpoints instead of training; see P1_addon.py ----
if os.environ.get("P1_ADDON") == "1":
    exec(open(os.environ.get("P1_ADDON_PATH", "/content/P1_addon.py")).read())
    sys.exit(0)

phase0_rows, pilot_logs = [], []
p0 = [p for p in PROTOCOLS if p["rep"] == "raw" and p["split"] == "group" and p["task"] in ("bin", "type")]
for pr in p0:
    for s in P["phase0_seeds"]:
        res, _, log = run_protocol(pr, s, "resnet18", P["phase0_epochs"], prefix="PHASE0_")
        phase0_rows += res; pilot_logs += log
p0df = pd.DataFrame(phase0_rows)
pilot = pd.DataFrame(pilot_logs); save_csv(pilot, SUB["logs"] / "pilot_curve.csv")
head = []
for (task, src), g in p0df.groupby(["task", "train_src"]):
    ins = g[g.test == "in_source"].ba.mean(); cr = g[g.test.str.startswith("cross_") & ~g.test.str.endswith("_n")].ba.mean()
    head.append({"task": task, "train_src": src, "in_source_ba": ins, "cross_ba": cr, "gap": ins - cr})
headdf = pd.DataFrame(head)
probe0 = []
for task, A, B in [("bin", "br35h", "sartaj"), ("bin", "sartaj", "br35h")]:
    for fname, feats in [("background_raw", BG_raw), ("format_raw", FMT)]:
        o, _ = probe_eval(feats, task, A, B)
        for r in o: probe0.append({"probe": fname, "train_src": A, **r})
probe0 = pd.DataFrame(probe0)
gap = float(headdf.gap.mean()) if len(headdf) else float("nan")
pilot_learns = bool(pilot.groupby(["protocol", "seed"]).val_ba.apply(lambda v: v.max() - v.iloc[0]).mean() > GATES["FLAT_LEARNING_EPS"]) if len(pilot) else False
DELTA_MIN = min(GATES["DELTA_MIN_DEFAULT"], gap / 2) if gap == gap and gap > 0 else GATES["DELTA_MIN_DEFAULT"]
H1_REACHABLE = (gap >= 2 * DELTA_MIN) and pilot_learns
GATE_CHANGELOG.append((time.strftime("%Y-%m-%d"), f"DELTA_MIN set to {DELTA_MIN:.3f}", f"Phase 0 gap {gap:.3f}", "pilot-informed"))
save_csv(headdf, SUB["tables"] / "feasibility_headroom.csv"); save_csv(probe0, SUB["tables"] / "feasibility_probes.csv")

def light(ok, amber=False): return "🟢" if ok else ("🟡" if amber else "🔴")
smoke = MODE == "SMOKE_TEST"
fe = ["# FEASIBILITY (Phase 0)", "",
      "| Question | Light | Why |", "|---|---|---|",
      f"| Can the paper claim a cross-source drop (H1) at Q1? | {'⚪ N/A at SMOKE resolution' if smoke else light(H1_REACHABLE, True)} | mean in-vs-cross BA gap = {gap:.3f}; Δmin = {DELTA_MIN:.3f}; pilot learns = {pilot_learns} |",
      f"| Should FULL_PAPER_RUN be launched? | {'⚪ N/A' if smoke else light(pilot_learns, True)} | {'confirmatory H1' if H1_REACHABLE else 'two-sided benchmark framing'} |",
      f"| Do non-anatomical probes carry label signal (H2 pilot)? | {'⚪ N/A' if smoke else light(bool((probe0[probe0.test=='in_source_groupedCV'].ba > GATES['PROBE_BA_MIN']).any()), True)} | max in-source probe BA = {probe0[probe0.test=='in_source_groupedCV'].ba.max():.3f} |",
      "", f"Overall: {'⚪ SMOKE — plumbing only' if smoke else (('🟢 FEASIBLE-Q1' if H1_REACHABLE else '🟡 FEASIBLE-Q2 (benchmark framing, two-sided H1)') if pilot_learns else '🔴 REDESIGN (flat learning)')}",
      "", "## Literature anchor", "Zhang (SSRN): matched vs unmatched test images 14.3 pts; Chhetri 2026: image vs patient split up to 3.71 pts; "
      "Khan 2026: 4.17 pts (confounded); Vangala 2026: test-only dedup −0.009, Nickparvar→cleaned SARTAJ −0.030 BA, header probe 0.959 BA.",
      "", "## Headroom", headdf.to_markdown(index=False) if len(headdf) else "none", "", "## Probes", probe0.to_markdown(index=False) if len(probe0) else "none",
      "", "## Model-fidelity note", "Sources differ in format, size and background (headers and padding), so a source mechanism exists by construction; "
      "the question is its size after deduplication.", "", f"## Gate changelog\n{GATE_CHANGELOG}"]
save_text("\n".join(fe), OUT / "FEASIBILITY.md")
print("\n".join(fe[:8]))
if not smoke and not H1_REACHABLE:
    print("\n" + "🔴" * 3 + " REFRAME-BEFORE-FULL: H1 not reachable at the pre-declared size; FULL continues as a two-sided benchmark. " + "🔴" * 3)

# %%
# ===================== MAIN RUNS (resumable) =====================
res_file = SUB["tables"] / f"per_seed_metrics_{MODE}.csv"
done = pd.read_csv(res_file) if res_file.exists() else pd.DataFrame()
if len(done):   # old ConvNeXt rows (pre-fix learning rate) are not reused
    _ver = done["ver"].fillna("") if "ver" in done.columns else pd.Series("", index=done.index)
    done = done[~(done.arch.astype(str).str.startswith("convnext") & (_ver != "p1-v1.3cnx"))]
all_res = [done] if len(done) else []
all_logs, futility_flag = [], False
for arch in P["archs"]:
    for si, seed in enumerate(P["seeds"]):
        if futility_flag and not FORCE_CONTINUE: break
        for pr in PROTOCOLS:
            if len(done) and ((done.protocol == pr["name"]) & (done.arch == arch) & (done.seed == seed)).any(): continue
            res, preds, log = run_protocol(pr, seed, arch, P["epochs"])
            all_res.append(pd.DataFrame(res)); all_logs += [dict(r, protocol=pr['name'], arch=arch, seed=seed) for r in log]
            preds.to_parquet(SUB["predictions"] / f"{pr['name']}_{arch}_s{seed}_{MODE}.parquet") if len(preds) else None
            save_csv(pd.concat(all_res), res_file)
            print(f"[{arch} s{seed}] {pr['name']}: " + ", ".join(f"{r['test']}={r['ba']:.3f}" for r in res))
        if si == 0 and MODE != "SMOKE_TEST":   # interim futility on H1 (first seed)
            cur = pd.concat(all_res); cur = cur[(cur.arch == arch) & (cur.split == "group") & (cur.rep == "raw")]
            g = cur[cur.test == "in_source"].groupby("protocol").ba.mean() - cur[cur.test.str.startswith("cross_")].groupby("protocol").ba.mean()
            if len(g.dropna()) and g.max() < DELTA_MIN:
                print(f"FUTILITY-WARNING: all first-seed in-vs-cross gaps < Δmin ({g.max():.3f} < {DELTA_MIN:.3f}). "
                      "Benchmark remains valid as two-sided; set FORCE_CONTINUE=True to run remaining seeds.")
                futility_flag = True
R = pd.concat(all_res).reset_index(drop=True)
R = R[~R.test.str.endswith("_n")]
R = R[R.arch.isin(P["archs"]) | ~R.arch.isin(["resnet18", "efficientnet_b0", "convnext_tiny"])]   # keep classical baselines; drop archs not in this run
save_csv(pd.DataFrame(all_logs), SUB["logs"] / f"training_curves_{MODE}.csv")

# Classical baselines (same splits): majority, logistic(histogram), LightGBM(histogram+gradient)
cls_rows = []
for pr in PROTOCOLS:
    feats = A_raw if pr["rep"] == "raw" else A_harm
    for seed in P["seeds"]:
        m, tr, va, tests = make_split(pr, seed)
        y_all = np.full(len(man), -1, dtype=np.int64); y_all[m.img_id.values] = m.y.values
        for B in pr["cross"]:
            mb = task_frame(pr["task"], B); y_all[mb.img_id.values] = mb.y.values
        trv = np.concatenate([tr, va]); nc = NCLS[pr["task"]]
        models_ = {"majority": None,
                   "logreg_hist": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                   "lgbm_hist_grad": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced", verbose=-1, random_state=seed)}
        for mn, clf in models_.items():
            if clf is not None: clf.fit(feats[trv], y_all[trv])
            for tname, idx in tests.items():
                if not isinstance(idx, np.ndarray) or len(idx) == 0: continue
                if clf is None:
                    prob = np.zeros((len(idx), nc)); prob[:, np.bincount(y_all[trv], minlength=nc).argmax()] = 1
                else:
                    pp = clf.predict_proba(feats[idx]); prob = np.zeros((len(idx), nc)); prob[:, clf.classes_] = pp
                cls_rows.append({"protocol": pr["name"], "task": pr["task"], "train_src": pr["train_src"], "rep": pr["rep"], "split": pr["split"],
                                 "arch": mn, "seed": seed, "test": tname, **metrics(y_all[idx], prob, nc)})
R = pd.concat([R, pd.DataFrame(cls_rows)], ignore_index=True)
save_csv(R, SUB["tables"] / "per_seed_metrics.csv")

# %%
# ===================== AGGREGATION, CIs, TESTS =====================
def cluster_boot_ba(pred_df, B, seed=0):
    """95% CI of seed-averaged BA, resampling groups (duplicate cluster / patient). Vectorised: multinomial group weights."""
    if pred_df.empty: return (np.nan, np.nan)
    g = man.set_index("img_id").group
    pdf = pred_df.assign(grp=g.loc[pred_df.img_id].values)
    groups, gi = np.unique(pdf.grp.values, return_inverse=True); G = len(groups)
    classes = np.unique(pdf.y.values); C = len(classes); ci = np.searchsorted(classes, pdf.y.values)
    seeds = np.unique(pdf.seed.values); si = np.searchsorted(seeds, pdf.seed.values); S = len(seeds)
    N = np.zeros((S, G, C)); TP = np.zeros((S, G, C))
    np.add.at(N, (si, gi, ci), 1); np.add.at(TP, (si, gi, ci), (pdf.y.values == pdf.pred.values).astype(float))
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)   # B x G
    bas = []
    for s_ in range(S):
        n = W @ N[s_]; tp = W @ TP[s_]                        # B x C
        with np.errstate(invalid="ignore", divide="ignore"): rec = np.where(n > 0, tp / n, np.nan)
        bas.append(np.nanmean(rec, axis=1))
    vals = np.nanmean(np.vstack(bas), axis=0)
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))

agg = (R.groupby(["protocol", "task", "train_src", "rep", "split", "arch", "test"])
         .agg(ba_mean=("ba", "mean"), ba_sd=("ba", "std"), f1_mean=("macro_f1", "mean"), auroc_mean=("auroc", "mean"),
              ece_mean=("ece", "mean"), n=("n", "first"), seeds=("seed", "nunique")).reset_index())
cis = []
for r in agg.itertuples():
    files = sorted(SUB["predictions"].glob(f"{r.protocol}_{r.arch}_s*_{MODE}.parquet"))
    if not files: cis.append((np.nan, np.nan)); continue
    pdf = pd.concat([pd.read_parquet(f) for f in files]); pdf = pdf[pdf.test == r.test]
    cis.append(cluster_boot_ba(pdf, P["boot"]))
agg["ba_ci_lo"], agg["ba_ci_hi"] = zip(*cis) if cis else ([], [])
save_csv(agg, SUB["tables"] / "protocol_results_agg.csv")

tests_rows = []
def paired(a, b, alternative):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2: return dict(diff=float(np.mean(d)) if len(d) else np.nan, t_p=np.nan, w_p=np.nan, d_z=np.nan, n=len(d))
    t = st.ttest_1samp(d, 0, alternative=alternative).pvalue
    try: w = st.wilcoxon(d, alternative=alternative, method="exact").pvalue
    except ValueError: w = np.nan
    return dict(diff=float(d.mean()), t_p=float(t), w_p=float(w), d_z=float(d.mean() / (d.std(ddof=1) + 1e-12)), n=len(d))

cnn = R[R.arch.isin(P["archs"])]
for (proto, arch), g in cnn[cnn.split == "group"].groupby(["protocol", "arch"]):
    ins = g[g.test == "in_source"].set_index("seed").ba
    for t in [x for x in g.test.unique() if x.startswith("cross_")]:
        cr = g[g.test == t].set_index("seed").ba; s = ins.index.intersection(cr.index)
        tests_rows.append({"hypothesis": "H1", "comparison": f"{proto} in_source - {t}", "arch": arch, **paired(ins[s], cr[s], "greater")})
for task, A in [("bin", "br35h"), ("bin", "sartaj"), ("type", "figshare"), ("type", "sartaj")]:
    for arch in P["archs"]:
        raw_ = cnn[(cnn.protocol == f"{task}_{A}_raw") & (cnn.arch == arch) & cnn.test.str.startswith("cross_")].set_index("seed").ba
        harm_ = cnn[(cnn.protocol == f"{task}_{A}_harm") & (cnn.arch == arch) & cnn.test.str.startswith("cross_")].set_index("seed").ba
        s = raw_.index.intersection(harm_.index)
        if len(s): tests_rows.append({"hypothesis": "H3", "comparison": f"{task}_{A} cross: harm - raw", "arch": arch, **paired(harm_[s], raw_[s], "two-sided")})
for arch in P["archs"]:
    img = cnn[(cnn.protocol == "type_figshare_raw_IMAGELEVEL") & (cnn.arch == arch) & (cnn.test == "in_source")].set_index("seed").ba
    pid = cnn[(cnn.protocol == "type_figshare_raw") & (cnn.arch == arch) & (cnn.test == "in_source")].set_index("seed").ba
    s = img.index.intersection(pid.index)
    if len(s): tests_rows.append({"hypothesis": "H4", "comparison": "figshare image-level - patient-disjoint", "arch": arch, **paired(img[s], pid[s], "greater")})
    off = cnn[(cnn.protocol == "four_nickparvar_raw_OFFICIAL") & (cnn.arch == arch) & (cnn.test == "in_source")].set_index("seed").ba
    clu = cnn[(cnn.protocol == "four_nickparvar_raw_CLUSTER") & (cnn.arch == arch) & (cnn.test == "in_source")].set_index("seed").ba
    s = off.index.intersection(clu.index)
    if len(s): tests_rows.append({"hypothesis": "H4b", "comparison": "nickparvar official - cluster-disjoint (retrained)", "arch": arch, **paired(off[s], clu[s], "greater")})
TT = pd.DataFrame(tests_rows)
if len(TT):
    pv = TT.t_p.fillna(TT.w_p).fillna(1.0).values   # paired t across seeds is primary; exact Wilcoxon (min p=1/2^n) reported as sensitivity
    TT["p_used"] = pv; TT["p_holm"] = multipletests(pv, method="holm")[1]
save_csv(TT, SUB["tables"] / "stats_tests.csv")
pvi = TT[TT.hypothesis.isin(["H4", "H4b"])] if len(TT) else TT
save_csv(pvi, SUB["tables"] / "patient_vs_image.csv")

# Probes (H2, H3): background raw / harmonised, format raw (harmonised format is constant by construction)
probe_rows = []
for task, A, B in [("bin", "br35h", "sartaj"), ("bin", "sartaj", "br35h"), ("type", "figshare", "sartaj"), ("type", "sartaj", "figshare")]:
    for fname, feats in [("background_raw", BG_raw), ("background_harm", BG_harm), ("format_raw", FMT)]:
        for seed in P["seeds"]:
            o, (ids, y, oof) = probe_eval(feats, task, A, B, seed)
            for r in o: probe_rows.append({"probe": fname, "task": task, "train_src": A, "seed": seed, **r})
PR = pd.DataFrame(probe_rows)
PRa = PR.groupby(["probe", "task", "train_src", "test"]).agg(ba_mean=("ba", "mean"), ba_sd=("ba", "std"), n=("n", "first")).reset_index()
chance = PRa.task.map({"bin": 0.5, "type": 1 / 3})
PRa["chance"] = chance
save_csv(PRa, SUB["tables"] / "probe_results.csv")

# %%
# ===================== SUBGROUPS, CALIBRATION, EFFICIENCY =====================
sg = []
for f in SUB["predictions"].glob(f"*_{MODE}.parquet"):
    d = pd.read_parquet(f); d = d.merge(man[["img_id", "source", "label", "width"]], on="img_id")
    d["size_tertile"] = pd.qcut(d.width.rank(method="first"), 3, labels=["small", "mid", "large"]) if len(d) >= 3 else "all"
    for key in ["label", "size_tertile"]:
        for (t, k), g in d.groupby(["test", key]):
            row = {"protocol": d.protocol.iloc[0], "arch": d.arch.iloc[0], "seed": d.seed.iloc[0], "test": t, "subgroup_var": key, "subgroup": str(k), "n": len(g)}
            row["acc"] = (g.pred == g.y).mean() if len(g) >= GATES["SUBGROUP_MIN_N"] else np.nan
            row["note"] = "" if len(g) >= GATES["SUBGROUP_MIN_N"] else "insufficient n"
            sg.append(row)
SG = pd.DataFrame(sg)
if len(SG):
    SG = SG.groupby(["protocol", "arch", "test", "subgroup_var", "subgroup"]).agg(n=("n", "first"), acc_mean=("acc", "mean"), note=("note", "first")).reset_index()
save_csv(SG, SUB["tables"] / "subgroups.csv")
save_csv(agg[["protocol", "arch", "test", "ece_mean"]], SUB["tables"] / "calibration.csv")
eff = R[R.arch.isin(P["archs"])].groupby("arch").agg(params=("n_params", "first"), train_sec_mean=("train_sec", "mean"), inf_ms_per_img=("inf_ms_per_img", "mean")).reset_index()
eff["hardware"] = torch.cuda.get_device_name(0) if DEVICE == "cuda" else platform.processor() or "cpu"
save_csv(eff, SUB["tables"] / "efficiency.csv")

# %%
# ===================== FIGURES (from saved tables only) =====================
def savefig(fig, name):
    for ext in ["pdf", "svg"]: fig.savefig(SUB["figures"] / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(SUB["figures"] / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("figure", name)

c = pd.read_csv(SUB["tables"] / "containment_matrix.csv")
if len(c):
    piv = c.pivot(index="container_A", columns="corpus_B", values="pct_B_in_A")
    fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(piv.values, cmap="Greys", vmin=0, vmax=100)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right"); ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v: ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > 50 else "black", fontsize=8)
    ax.set_xlabel("Corpus B"); ax.set_ylabel("Container corpus A"); fig.colorbar(im, label="% of B with a near-duplicate in A"); savefig(fig, "fig2_containment")
sw = pd.read_csv(SUB["tables"] / "ncc_threshold_sweep.csv")
fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(sw.threshold, sw.images_in_multi_clusters, "k-o"); ax.set_xlabel("NCC threshold"); ax.set_ylabel("Images in duplicate clusters"); savefig(fig, "fig_s1_ncc_sweep")
a = pd.read_csv(SUB["tables"] / "protocol_results_agg.csv")
a = a[(a.split == "group") & a.arch.isin(P["archs"])]
if len(a):
    fig, ax = plt.subplots(figsize=(7, 3.5)); labs, x = [], 0
    for (proto, arch), g in a.groupby(["protocol", "arch"]):
        for t, mk in [("in_source", "o"), ]:
            pass
        ins = g[g.test == "in_source"]; crs = g[g.test.str.startswith("cross_")]
        if len(ins) and len(crs):
            ax.errorbar([x - 0.1], ins.ba_mean, yerr=ins.ba_sd.fillna(0), fmt="ko"); ax.errorbar([x + 0.1], crs.ba_mean, yerr=crs.ba_sd.fillna(0), fmt="ks", mfc="white")
            labs.append(f"{proto}\n{arch}"); x += 1
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6); ax.set_ylabel("Balanced accuracy"); ax.set_title("● in-source  ○ cross-source")
    savefig(fig, "fig3_in_vs_cross")
pr_ = pd.read_csv(SUB["tables"] / "probe_results.csv")
if len(pr_):
    fig, ax = plt.subplots(figsize=(7, 3)); lab = pr_.probe + "|" + pr_.task + "|" + pr_.train_src + "|" + pr_.test
    ax.bar(range(len(pr_)), pr_.ba_mean, color="grey", yerr=pr_.ba_sd.fillna(0)); ax.plot(range(len(pr_)), pr_.chance, "k_", ms=12)
    ax.set_xticks(range(len(pr_))); ax.set_xticklabels(lab, rotation=75, ha="right", fontsize=5); ax.set_ylabel("Probe balanced accuracy"); savefig(fig, "fig4_probes")
pc = pd.read_csv(SUB["logs"] / "pilot_curve.csv")
if len(pc):
    fig, ax = plt.subplots(figsize=(4, 3))
    for (p_, s_), g in pc.groupby(["protocol", "seed"]): ax.plot(g.epoch, g.val_ba, color="k", alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation BA"); savefig(fig, "fig_s2_pilot_curve")

# %%
# ===================== GATE DASHBOARD, RESULTS_AUDIT, PLAIN-LANGUAGE SUMMARY =====================
TT = pd.read_csv(SUB["tables"] / "stats_tests.csv") if (SUB["tables"] / "stats_tests.csv").stat().st_size > 5 else pd.DataFrame()
def hyp_status(h, want_positive=True, min_diff=0.0):
    if TT.empty or h not in set(TT.hypothesis): return "NOT RUN", None
    t = TT[TT.hypothesis == h]
    ok = ((t.p_holm < GATES["ALPHA"]) & (t["diff"] >= min_diff)).mean()
    return ("PASS" if ok >= 0.5 else ("PARTIAL" if ok > 0 else "FAIL")), t
h1, _ = hyp_status("H1", min_diff=DELTA_MIN)
h3, t3 = hyp_status("H3")
h4, _ = hyp_status("H4")
prb = PRa[(PRa.task == "bin") & (PRa.probe != "background_harm") & (PRa.test == "in_source_groupedCV")]
h2 = "PASS" if len(prb) and (prb.ba_mean > GATES["PROBE_BA_MIN"]).any() else "FAIL"
if MODE == "SMOKE_TEST":   # resolution-limited checks are not judged in SMOKE
    h1 = h2 = h3 = h4 = "N/A at SMOKE resolution"
learning = R[R.arch.isin(P["archs"])].learning.value_counts().to_dict()
is_pub = (MODE == "FULL_PAPER_RUN") and not LOCAL_DUMMY
def L(s): return {"PASS": "🟢", "PARTIAL": "🟡", "FAIL": "🔴", "NOT RUN": "⚪"}.get(s, "⚪")
if MODE == "SMOKE_TEST":
    overall = "⚪ SMOKE-PASS — plumbing works; numbers are meaningless (NOT FOR PUBLICATION)"
    launch = "N/A"
elif MODE == "STANDARD_RUN":
    _nflat = learning.get("FLAT-LEARNING", 0); _nrun = max(1, sum(learning.values()))
    _learn_ok = _nflat / _nrun <= 0.10
    go = _learn_ok and h1 in ("PASS", "PARTIAL")
    launch = "LAUNCH-FULL GO (confirmatory H1)" if go else ("LAUNCH-FULL GO-BENCHMARK (two-sided H1)" if _learn_ok else "LAUNCH-FULL NO-GO — fix learning first")
    overall = ("🟢 " if go else "🟡 ") + launch
else:
    n_pass = sum(s == "PASS" for s in [h1, h2, h3, h4])
    overall = "🟢 GO-Q1-CANDIDATE" if n_pass >= 3 and h1 == "PASS" else ("🟡 GO-Q2-CANDIDATE (benchmark framing)" if n_pass >= 1 else "🔴 IMPROVE / STOP-REDESIGN")
    launch = "N/A (FULL done)"
dash = ["| Question | Light | Why |", "|---|---|---|",
        f"| H1: does accuracy drop when testing on another source? | {L(h1)} | {h1}; Δmin = {DELTA_MIN:.3f} |",
        f"| H2: do background/format cues alone predict the label? | {L(h2)} | max in-source probe BA = {prb.ba_mean.max() if len(prb) else float('nan'):.3f} (threshold {GATES['PROBE_BA_MIN']}) |",
        f"| H3: does harmonisation change cross-source accuracy? | {L(h3)} | two-sided, Holm-corrected |",
        f"| H4: does image-level splitting inflate Figshare accuracy? | {L(h4)} | paired over seeds, one-sided |",
        f"| Learning diagnostics | {'🟢' if learning.get('FLAT-LEARNING',0)==0 else '🔴'} | {learning} |",
        f"| Manuscript eligibility | {'🟢' if is_pub else '🔴'} | {'FULL_PAPER_RUN on real data' if is_pub else 'not FULL / dummy data'} |",
        "", f"Overall: {overall}"]
plain = ["## Plain-language summary"]
if not TT.empty:
    for r in TT.itertuples():
        word = "higher" if r.diff > 0 else "lower"
        plain.append(f"- {r.hypothesis} {r.comparison} [{r.arch}]: difference {r.diff:+.3f} BA ({word}), p(Holm) = {r.p_holm:.3g}, d_z = {r.d_z:.2f}, n seeds = {r.n}.")
claims = pd.DataFrame([
    {"claim": "Containment between corpora", "slot": "R2", "file": "tables/containment_matrix.csv", "value": f"max {cont.pct_B_in_A.max():.1f}%" if len(cont) else "NA", "gate": "PASS" if len(cont) else "FAIL"},
    {"claim": "In vs cross-source gap (H1)", "slot": "R3/R4", "file": "tables/protocol_results_agg.csv; stats_tests.csv", "value": f"Phase0 gap {gap:.3f}", "gate": h1},
    {"claim": "Non-anatomical probes (H2)", "slot": "R5/R6", "file": "tables/probe_results.csv", "value": f"{prb.ba_mean.max() if len(prb) else float('nan'):.3f}", "gate": h2},
    {"claim": "Harmonisation effect (H3)", "slot": "R6", "file": "tables/stats_tests.csv", "value": "see H3 rows", "gate": h3},
    {"claim": "Patient-leakage inflation (H4)", "slot": "R7", "file": "tables/patient_vs_image.csv", "value": "see H4 rows", "gate": h4},
])
nxt = pd.DataFrame([
    {"item": "feasibility", "status": "H1 reachable" if H1_REACHABLE else "two-sided", "evidence": "FEASIBILITY.md", "next": "launch FULL" if MODE == "STANDARD_RUN" else "-", "cost_h": 7 if MODE == "STANDARD_RUN" else 0},
    {"item": "learning", "status": str(learning), "evidence": "logs/training_curves_*.csv", "next": "fix if FLAT", "cost_h": 0.5},
    {"item": "futility", "status": "WARNING" if futility_flag else "OK", "evidence": "console", "next": "FORCE_CONTINUE if benchmark", "cost_h": 0},
    {"item": "novelty check", "status": "dated search done 2026-11-07; re-check in Stage 2", "evidence": "STAGE1_P1_DESIGN.md §2", "next": "Stage 2 re-check", "cost_h": 0.5},
])
audit = ["# RESULTS_AUDIT — P1", f"Mode {MODE} | account {RUN_ACCOUNT} | {time.ctime()} | LOCAL_DUMMY={LOCAL_DUMMY}",
         "**NOT FOR PUBLICATION**" if not is_pub else "Eligible for manuscript slots (FULL, real data).", ""] + dash + ["", "## Claim-to-evidence", claims.to_markdown(index=False),
         "", "## Next actions", nxt.to_markdown(index=False), "", *plain, "", f"LAUNCH-FULL: {launch}", f"Gate changelog: {GATE_CHANGELOG}"]
save_text("\n".join(audit), OUT / "RESULTS_AUDIT.md")
print("\n".join(dash))
print("\n".join(plain[:12]))

# %%
# ===================== MANIFEST, ENV LOCK, ZIP =====================
run_manifest = {"mode": MODE, "account": RUN_ACCOUNT, "datasets_confirmed": DATASETS_CONFIRMED, "model_version": MODEL_VERSION,
                "split_seed": SPLIT_SEED, "profile": P, "gates": GATES, "delta_min": DELTA_MIN, "gate_changelog": GATE_CHANGELOG,
                "device": DEVICE, "python": sys.version, "time": time.ctime(), "local_dummy": LOCAL_DUMMY, "n_images": int(len(man))}
save_text(json.dumps(run_manifest, indent=2, default=str), SUB["config"] / "run_manifest.json")
env = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
save_text(env, SUB["config"] / "env_lock.txt")
required = ["tables/corpus_counts.csv", "tables/containment_matrix.csv", "tables/split_table.csv", "tables/per_seed_metrics.csv",
            "tables/protocol_results_agg.csv", "tables/stats_tests.csv", "tables/probe_results.csv", "tables/subgroups.csv",
            "tables/calibration.csv", "tables/efficiency.csv", "tables/feasibility_headroom.csv", "logs/pilot_curve.csv",
            "data_audit/DATA_PROVENANCE.md", "config/run_manifest.json", "config/env_lock.txt"]
missing = [f for f in required if not (OUT / f).exists() or (OUT / f).stat().st_size == 0]
for f in ["FEASIBILITY.md", "RESULTS_AUDIT.md"]:
    if not (OUT / f).exists(): missing.append(f)
print("MISSING OUTPUTS:", missing if missing else "none")
zp = ROOT / f"P1_results_{MODE}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob("*"):
        if f.is_file() and "checkpoints" not in f.parts: z.write(f, f.relative_to(ROOT))
print(f"ZIP: {zp} ({os.path.getsize(zp)/1e6:.1f} MB)")
assert not missing, f"Missing outputs block claims: {missing}"

# %% [markdown]
# ## Limitations (for the manuscript's threats-to-validity section)
# - Patient identifiers exist only for Figshare; Kaggle images inherit them only when they match a Figshare slice. Patient overlap in SARTAJ/Br35H outside those matches cannot be measured.
# - SARTAJ glioma labels are reported to be noisy; tumour-type results are also given without glioma.
# - All analyses use 2D slices; scanner, sequence parameters and demographics are unavailable.
# - Harmonisation is one canonical pipeline; other harmonisation methods may behave differently.
