# P1 ADD-ON (approved 8 Nov 2026, Step 5b): R5 attribution share + R7 Causal Cue Intervention Test (CCIT).
# Executed inside P1_pipeline.py's namespace (P1_ADDON=1) after the shared data, cache, clusters and splits are rebuilt,
# so every split and checkpoint is exactly the FULL_PAPER_RUN one. No model is trained here.
import io, zipfile, itertools
from statsmodels.stats.multitest import multipletests
import scipy.stats as _st

AO = OUT / "addon"; (AO / "tables").mkdir(parents=True, exist_ok=True); (AO / "figures").mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "ADDON_STATUS.md"
def astatus(msg):
    line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"; print(line)
    with open(STATUS, "a") as f: f.write(line + "\n")

if not LOCAL_DUMMY and not (ROOT / "P1_results_FULL_PAPER_RUN.zip").exists():
    raise RuntimeError("FULL_PAPER_RUN is not finished (no P1_results_FULL_PAPER_RUN.zip). Fix: let the FULL run finish first, "
                       "then Run all on this add-on notebook. Never run two GPU notebooks on this account at once (lesson L14).")

ARCHS = P["archs"]; SEEDS = P["seeds"]
RAW_PROTOS = [p for p in PROTOCOLS if p["rep"] == "raw" and p["split"] == "group" and p["task"] in ("bin", "type")]
N_PERM = 200 if LOCAL_DUMMY else 1000
CAP_ATTR = 40 if LOCAL_DUMMY else 200           # images per model for attribution
rng_global = np.random.default_rng(2026)

def load_model(pr, arch, seed):
    ver = "p1-v1.3cnx" if arch.startswith("convnext") else MODEL_VERSION
    tag = f"{pr['name']}_{arch}_s{seed}_{MODE}_{ver}_{SPLIT_SEED}"
    ck = SUB["checkpoints"] / f"{tag}.pt"
    if not ck.exists(): return None
    try: state = torch.load(ck, map_location=DEVICE)
    except Exception: state = torch.load(ck, map_location=DEVICE, weights_only=False)   # own checkpoint (L11)
    m = timm.create_model(arch, pretrained=False, num_classes=NCLS[pr["task"]]).to(DEVICE)
    m.load_state_dict(state["model"]); m.eval(); return m

def labels_for(pr):
    y = np.full(len(man), -1, dtype=np.int64)
    for s in [pr["train_src"]] + pr["cross"]:
        mb = task_frame(pr["task"], s); y[mb.img_id.values] = mb.y.values
    return y

def predict_arr(model, A):          # A: (n,224,224) uint8
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
        for s in range(0, len(A), 128):
            x = torch.from_numpy(A[s:s + 128]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
            out.append(torch.softmax(model(((x - MEAN) / STD).to(DEVICE)).float(), 1).cpu().numpy())
    return np.concatenate(out)

MASKS = {}
def bmask(i):
    if i not in MASKS: MASKS[i] = brain_mask(X_raw[i])
    return MASKS[i]

# ---------------- interventions (anatomy pixels are never changed) ----------------
def bg_transplant(target_ids, donor_pool, rng):
    """Keep the target's brain region; replace everything outside it with a donor image's background."""
    out = X_raw[target_ids].copy()
    donors = rng.choice(donor_pool, size=len(target_ids), replace=True)
    for k, (t, d) in enumerate(zip(target_ids, donors)):
        mt, md = bmask(t), bmask(d); dimg = X_raw[d]
        fill = np.median(dimg[~md]) if (~md).any() else 0
        bg = np.where(md, fill, dimg).astype(np.uint8)
        out[k] = np.where(mt, X_raw[t], bg)
    return out

def zero_outside(target_ids):
    out = X_raw[target_ids].copy()
    for k, t in enumerate(target_ids): out[k][~bmask(t)] = 0
    return out

def reencode(target_ids, donor_pool, rng):
    """Re-save the ORIGINAL file at a donor file's size and JPEG tables (or PNG if the donor is PNG), then the same raw resize."""
    out = np.zeros((len(target_ids), IMG, IMG), np.uint8)
    donors = rng.choice(donor_pool, size=len(target_ids), replace=True)
    for k, (t, d) in enumerate(zip(target_ids, donors)):
        a = load_gray(man.path.iloc[t])
        with Image.open(man.path.iloc[d]) as dim:
            W, H, q, fmt = dim.width, dim.height, getattr(dim, "quantization", None), (dim.format or "PNG")
        im = Image.fromarray(a).resize((max(W, 8), max(H, 8)), Image.BILINEAR); buf = io.BytesIO()
        if fmt.upper() in ("JPEG", "JPG") and q: im.save(buf, "JPEG", qtables=q)
        else: im.save(buf, "PNG")
        buf.seek(0)
        with Image.open(buf) as rim: out[k] = raw_repr(np.array(rim.convert("L")))
    return out

def perm_test(d, n_perm, rng):
    """Paired sign-flip permutation on per-image differences; empirical p = (r+1)/(n+1), two-sided."""
    d = np.asarray(d, float); obs = abs(d.mean())
    if len(d) == 0 or np.allclose(d, 0): return 1.0
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    r = int((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum())
    return (r + 1) / (n_perm + 1)

# ---------------- R7: Causal Cue Intervention Test ----------------
# Experiments (each = intervention vs matched control; only non-anatomical pixels or file encoding change):
#  T1 cross-source test images get a TRAINING-source background   vs control: a background from their own (cross) source
#  T2 in-source test images get a CROSS-source background          vs control: a background from their own (training) source
#  T3 in-source test files re-encoded with CROSS-source size/tables vs control: re-encoded with own-source size/tables
#  T4 outside-brain pixels set to zero (in-source and cross-source) vs control: unedited
def ccit_one(pr, seed, models, rng, tests_subset=("T1", "T2", "T3", "T4")):
    _, tr, va, tests = make_split(pr, seed); y = labels_for(pr)
    A, B = pr["train_src"], pr["cross"][0]
    te_in, te_x = tests["in_source"], tests[f"cross_{B}"]
    pool_A = np.concatenate([tr, va]); pool_B = te_x       # donors: training-source train/val images; cross-source images
    conds = {}
    if "T1" in tests_subset:
        conds["T1"] = (te_x, bg_transplant(te_x, pool_A, rng), bg_transplant(te_x, pool_B, rng))
    if "T2" in tests_subset:
        conds["T2"] = (te_in, bg_transplant(te_in, pool_B, rng), bg_transplant(te_in, pool_A, rng))
    if "T3" in tests_subset:
        conds["T3"] = (te_in, reencode(te_in, pool_B, rng), reencode(te_in, pool_A, rng))
    if "T4" in tests_subset:
        conds["T4_in"] = (te_in, zero_outside(te_in), X_raw[te_in])
        conds["T4_cross"] = (te_x, zero_outside(te_x), X_raw[te_x])
    rows, perimg = [], []
    for arch, model in models.items():
        for cname, (ids, Xe, Xc) in conds.items():
            pe, pc = predict_arr(model, Xe), predict_arr(model, Xc); yy = y[ids]
            be, bc = balanced_accuracy_score(yy, pe.argmax(1)), balanced_accuracy_score(yy, pc.argmax(1))
            dtrue = pe[np.arange(len(yy)), yy] - pc[np.arange(len(yy)), yy]
            flip = float((pe.argmax(1) != pc.argmax(1)).mean())
            rows.append({"protocol": pr["name"], "arch": arch, "seed": seed, "test": cname, "n": len(ids),
                         "ba_intervention": be, "ba_control": bc, "delta_ba": be - bc,
                         "mean_delta_p_true": float(dtrue.mean()), "flip_rate": flip})
            perimg.append(pd.DataFrame({"protocol": pr["name"], "arch": arch, "seed": seed, "test": cname,
                                        "img_id": ids, "d_p_true": dtrue}))
    return rows, perimg

astatus("ADD-ON START (R7 CCIT pilot -> R7 full -> R5 attribution)")

# ---- pilot first (approved risk check): 1 model x 1 direction ----
pr0 = next(p for p in RAW_PROTOS if p["name"] == "bin_sartaj_raw")
m0 = load_model(pr0, ARCHS[0], SEEDS[0])
if m0 is None: raise RuntimeError(f"Missing FULL checkpoint for {pr0['name']} {ARCHS[0]} s{SEEDS[0]}. Fix: finish FULL_PAPER_RUN first.")
pil_rows, pil_img = ccit_one(pr0, SEEDS[0], {ARCHS[0]: m0}, np.random.default_rng(0), ("T1", "T2"))
pil = pd.DataFrame(pil_rows); save_csv(pil, AO / "tables" / "ccit_pilot.csv")
pil_eff = pil.delta_ba.abs().max()
pil_light = "🟢" if pil_eff >= 0.05 else ("🟡" if pil_eff >= 0.02 else "🔴")
astatus(f"PILOT {pil_light} CCIT bin_sartaj_raw {ARCHS[0]} s0: " + "; ".join(f"{r.test} ΔBA={r.delta_ba:+.3f} flip={r.flip_rate:.2f}" for r in pil.itertuples())
        + " | 🟢 ≥0.05, 🟡 0.02–0.05, 🔴 <0.02 (small effect: report honestly, paper reframes as benchmark + audit)")
del m0; torch.cuda.empty_cache()

# ---- full CCIT ----
ccit_rows, ccit_img = [], []
PART = AO / "tables" / "partial"; PART.mkdir(parents=True, exist_ok=True)
import pickle as _pk
for pr in RAW_PROTOS:
    pf = PART / f"ccit_{pr['name']}.pkl"
    if pf.exists():                                   # resume: protocol already finished before a disconnect
        r, im = _pk.load(open(pf, "rb")); ccit_rows += r; ccit_img += im; astatus(f"CCIT resume {pr['name']} (loaded)"); continue
    _r0, _i0 = len(ccit_rows), len(ccit_img)
    for seed in SEEDS:
        models = {a: m for a in ARCHS if (m := load_model(pr, a, seed)) is not None}
        if not models: continue
        r, im = ccit_one(pr, seed, models, np.random.default_rng(1000 + seed))
        ccit_rows += r; ccit_img += im
        del models; torch.cuda.empty_cache()
    _pk.dump((ccit_rows[_r0:], ccit_img[_i0:]), open(pf, "wb"))
    astatus(f"CCIT done {pr['name']}")
C = pd.DataFrame(ccit_rows); save_csv(C, AO / "tables" / "ccit_per_seed.csv")
CI = pd.concat(ccit_img); CI.to_parquet(AO / "tables" / "ccit_per_image.parquet")

# statistics (L7): per protocol x arch x test: mean ΔBA across seeds, paired t over seeds (primary), bootstrap CI over seeds,
# sign-flip permutation on pooled per-image Δp(true class) with n=1000 (DeGrave-style empirical p), Holm over the family.
srows, prng = [], np.random.default_rng(7)
for (p_, a_, t_), g in C.groupby(["protocol", "arch", "test"]):
    d = g.delta_ba.values
    tp = _st.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else np.nan
    bs = [rng_global.choice(d, len(d)).mean() for _ in range(2000)]
    di = CI[(CI.protocol == p_) & (CI.arch == a_) & (CI.test == t_)].d_p_true.values
    srows.append({"protocol": p_, "arch": a_, "test": t_, "n_seeds": len(d), "mean_delta_ba": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                  "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "t_p": tp, "d_z": d.mean() / (d.std(ddof=1) + 1e-12) if len(d) > 1 else np.nan,
                  "mean_delta_p_true": di.mean(), "perm_p": perm_test(di, N_PERM, prng), "mean_flip_rate": g.flip_rate.mean()})
S = pd.DataFrame(srows)
ok = S.t_p.notna()
S.loc[ok, "t_p_holm"] = multipletests(S.loc[ok, "t_p"], method="holm")[1]
S["perm_p_holm"] = multipletests(S.perm_p, method="holm")[1]
save_csv(S, AO / "tables" / "ccit_summary.csv")

# ---------------- R5: attribution share (Grad-CAM + Integrated Gradients) ----------------
def gradcam(model, x, cls):
    feats = model.forward_features(x); feats.retain_grad()
    logits = model.forward_head(feats)
    logits[torch.arange(len(cls)), cls].sum().backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    return F.interpolate(cam, size=(IMG, IMG), mode="bilinear", align_corners=False)[:, 0].detach().cpu().numpy()

def intgrad(model, x, cls, steps=16):
    base = torch.zeros_like(x); tot = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi)[torch.arange(len(cls)), cls].sum()
        tot += torch.autograd.grad(out, xi)[0]
    return ((x - base) * tot / steps).abs().sum(1).detach().cpu().numpy()

def attribution(model, ids, regions_fn):
    rows = []
    for s in range(0, len(ids), 16):
        b = ids[s:s + 16]
        x = to_tensor(X_raw, b).to(DEVICE)
        with torch.no_grad(): cls = model(x).argmax(1)
        with torch.enable_grad():
            cams = gradcam(model, x.clone(), cls); model.zero_grad(set_to_none=True)
            igs = intgrad(model, x.clone(), cls); model.zero_grad(set_to_none=True)
        for k, i in enumerate(b):
            regs = regions_fn(i)
            for meth, amap in (("gradcam", cams[k]), ("intgrad", igs[k])):
                tot = amap.sum() + 1e-12
                rows.append({"img_id": int(i), "method": meth, **{f"share_{r}": float(amap[m].sum() / tot) for r, m in regs.items()},
                             **{f"area_{r}": float(m.mean()) for r, m in regs.items()}})
    return rows

def regions_brain(i):
    m = bmask(i); return {"brain": m, "background": ~m}

# BRISC tumour masks (segmentation_task/*/masks), matched to classification images by file stem
IMGX = (".png", ".jpg", ".jpeg", ".tif")
seg = [p for p in (DATA / "brisc").rglob("*") if p.suffix.lower() in IMGX and "segmentation" in str(p).lower()]
mask_files = {p.stem: p for p in seg if "mask" in str(p.parent).lower()}
seg_imgs = [p for p in seg if "mask" not in str(p.parent).lower()]
# classification and segmentation folders hold separate copies: match by file stem, then by identical bytes (md5)
md5_to_stem = {hashlib.md5(open(p, "rb").read()).hexdigest(): p.stem for p in seg_imgs if p.stem in mask_files}
br = man[man.source == "brisc"].copy(); br["stem"] = br.path.map(lambda p: Path(p).stem)
br["stem"] = [s_ if s_ in mask_files else md5_to_stem.get(h_, s_) for s_, h_ in zip(br.stem, br.md5)]
astatus(f"BRISC segmentation files found: {len(mask_files)} masks, {len(seg_imgs)} images")
br = br[br.stem.isin(mask_files) & br.label.isin(TYPES)]
astatus(f"BRISC tumour masks matched to classification images: {len(br)}")
TM = {}
def regions_tumour(i):
    if i not in TM:
        mk = np.array(Image.open(mask_files[br.loc[br.img_id == i, "stem"].iloc[0]]).convert("L").resize((IMG, IMG), Image.NEAREST)) > 127
        TM[i] = mk
    t = TM[i]; b = bmask(i)
    return {"tumour": t, "brain_non_tumour": b & ~t, "background": ~b & ~t}

attr_rows = []
for pr in RAW_PROTOS:
    for seed in SEEDS[:3]:
        _, tr, va, tests = make_split(pr, seed); used = set(man.group.iloc[np.concatenate([tr, va])])
        for arch in ARCHS:
            model = load_model(pr, arch, seed)
            if model is None: continue
            for tname in ["in_source", f"cross_{pr['cross'][0]}"]:
                ids = tests[tname]; ids = rng_global.choice(ids, min(CAP_ATTR, len(ids)), replace=False)
                for r in attribution(model, ids, regions_brain):
                    attr_rows.append({"protocol": pr["name"], "arch": arch, "seed": seed, "set": tname, "regions": "brain_vs_background", **r})
            if pr["task"] == "type" and len(br):
                bids = br[~br.group.isin(used)].img_id.values      # BRISC tumour images never seen in training/validation
                bids = rng_global.choice(bids, min(CAP_ATTR, len(bids)), replace=False) if len(bids) else bids
                for r in attribution(model, bids, regions_tumour):
                    attr_rows.append({"protocol": pr["name"], "arch": arch, "seed": seed, "set": "brisc_masked", "regions": "tumour_brain_background", **r})
            del model; torch.cuda.empty_cache()
    astatus(f"attribution done {pr['name']}")
AT = pd.DataFrame(attr_rows); save_csv(AT, AO / "tables" / "attribution_per_image.csv")
share_cols = [c for c in AT.columns if c.startswith("share_")]
ATS = AT.groupby(["protocol", "arch", "set", "regions", "method"])[share_cols].mean().reset_index()
save_csv(ATS, AO / "tables" / "attribution_summary.csv")

# ---------------- figures from saved tables ----------------
fig, ax = plt.subplots(figsize=(9, 4))
piv = S.groupby(["test", "arch"]).mean_delta_ba.mean().unstack()
piv.plot.bar(ax=ax); ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("ΔBA (intervention − control)")
ax.set_title("CCIT: effect of editing only non-anatomical cues"); fig.tight_layout(); fig.savefig(AO / "figures" / "ccit_delta_ba.png", dpi=200); plt.close(fig)
if len(ATS):
    sub = ATS[(ATS.regions == "brain_vs_background") & (ATS.method == "gradcam")]
    fig, ax = plt.subplots(figsize=(9, 4))
    sub.assign(key=sub.protocol + "|" + sub.set).groupby(["key", "arch"]).share_background.mean().unstack().plot.bar(ax=ax)
    ax.set_ylabel("share of Grad-CAM on background"); fig.tight_layout(); fig.savefig(AO / "figures" / "attribution_background_share.png", dpi=200); plt.close(fig)

# ---------------- traffic-light summary ----------------
def light_row(test):
    s = S[S.test == test]
    if not len(s): return "⚪", "not run"
    sig = (s.perm_p_holm < 0.05).mean(); eff = s.mean_delta_ba.abs().median()
    return ("🟢" if (sig >= 0.5 and eff >= 0.05) else "🟡" if (sig > 0 or eff >= 0.02) else "🔴"), f"median |ΔBA| {eff:.3f}; {100*sig:.0f}% of protocol×arch significant after Holm"
lines = ["# ADD-ON RESULTS — P1 (R5, R7)", f"Mode {MODE} | {time.ctime()} | LOCAL_DUMMY={LOCAL_DUMMY}", "",
         "| Question | Light | Evidence |", "|---|---|---|"]
for t, q in [("T1", "Does giving cross-source images a training-source background change accuracy?"),
             ("T2", "Does giving in-source images a cross-source background change accuracy?"),
             ("T3", "Does re-encoding files in the other source's format change accuracy?"),
             ("T4_in", "Does removing everything outside the brain change in-source accuracy?"),
             ("T4_cross", "Does removing everything outside the brain change cross-source accuracy?")]:
    l, e = light_row(t); lines.append(f"| {q} | {l} | {e} |")
if len(ATS):
    bgs = ATS[(ATS.regions == "brain_vs_background") & (ATS.method == "gradcam")].share_background.mean()
    lines.append(f"| How much Grad-CAM attention falls outside the brain? | {'🔴' if bgs >= 0.3 else '🟡' if bgs >= 0.15 else '🟢'} | mean share {bgs:.2f} |")
lines += ["", f"Pilot: {pil_light} (max |ΔBA| {pil_eff:.3f})", "", "## CCIT summary", S.round(4).to_markdown(index=False),
          "", "## Attribution summary", ATS.round(3).to_markdown(index=False) if len(ATS) else "none"]
save_text("\n".join(lines), AO / "ADDON_RESULTS.md")
zp = ROOT / "P1_addon_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in AO.rglob("*"):
        if f.is_file(): zf.write(f, f.relative_to(ROOT))
astatus(f"ADD-ON COMPLETE -> {zp.name}")
