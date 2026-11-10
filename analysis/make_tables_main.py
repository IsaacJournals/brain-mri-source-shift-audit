"""Manuscript tables built directly from the per-seed results (Tables 4, 10, 11, 13, 14, 15, 18, 19, 21, Supplementary Tables S1, S2)."""
import os, pandas as pd, numpy as np
F = os.environ.get("P1_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")) + "/"
d = pd.read_csv(F + "per_seed_metrics_FULL.csv")
A = {"resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-T"}
def ms(v): return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
out = {}
m = pd.read_csv(F + "manifest_with_clusters.csv")
ct = m.groupby(['source', 'official_split', 'label']).size().unstack(fill_value=0)
rows = ["| Source | Split | Glioma | Meningioma | Pituitary | No tumour | Tumour (unspecified) | Total |", "|---|---|---|---|---|---|---|---|"]
names = {"figshare": "Figshare", "sartaj": "SARTAJ", "br35h": "Br35H", "nickparvar": "Nickparvar", "brisc": "BRISC"}
for (s, sp), r in ct.iterrows():
    g = lambda k: int(r.get(k, 0))
    rows.append(f"| {names[s]} | {sp} | {g('glioma')} | {g('meningioma')} | {g('pituitary')} | {g('notumor')} | {g('tumor_unspec')} | {int(r.sum())} |")
rows.append(f"| **All** |  |  |  |  |  |  | **{len(m)}** |")
out["T_counts"] = "\n".join(rows)
pairs = [("bin_br35h", "cross_sartaj", "Br35H → SARTAJ"), ("bin_sartaj", "cross_br35h", "SARTAJ → Br35H"), ("type_figshare", "cross_sartaj", "Figshare → SARTAJ"), ("type_sartaj", "cross_figshare", "SARTAJ → Figshare")]
rows = ["| Direction | Representation | Architecture | In-source BA | New-source BA | Macro-F1 (new) | AUROC (new) | ECE (new) | n (new) |", "|---|---|---|---|---|---|---|---|---|"]
for p, ct_, lab in pairs:
    for rep in ["raw", "harm"]:
        for a in A:
            g = d[(d.protocol == f"{p}_{rep}") & (d.arch == a)]
            i = g[g.test == "in_source"]; c = g[g.test == ct_]
            rows.append(f"| {lab} | {'raw' if rep == 'raw' else 'harmonised'} | {A[a]} | {ms(i.ba)} | {ms(c.ba)} | {c.macro_f1.mean():.3f} | {c.auroc.mean():.3f} | {c.ece.mean():.3f} | {int(c.n.mean())} |")
out["T_cross"] = "\n".join(rows)
rows = ["| Protocol | Test | Architecture | BA | Glioma | Meningioma | Pituitary | No tumour | n |", "|---|---|---|---|---|---|---|---|---|"]
for p, lab in [("four_nickparvar_raw_OFFICIAL", "Official split"), ("four_nickparvar_raw_CLUSTER", "Cluster-disjoint split")]:
    for t in ["in_source", "official_test_clean"]:
        for a in A:
            g = d[(d.protocol == p) & (d.test == t) & (d.arch == a)]
            if len(g) == 0: continue
            tl = {"in_source": "in-source" if "CLUSTER" in p else "official test", "official_test_clean": "duplicate-free test"}[t]
            rows.append(f"| {lab} | {tl} | {A[a]} | {ms(g.ba)} | {g.recall_c0.mean():.3f} | {g.recall_c1.mean():.3f} | {g.recall_c2.mean():.3f} | {g.recall_c3.mean():.3f} | {int(g.n.mean())} |")
out["T_perclass"] = "\n".join(rows)
rows = ["| Architecture | Image-level split BA | Patient-disjoint split BA | Difference |", "|---|---|---|---|"]
for a in A:
    im = d[(d.protocol == "type_figshare_raw_IMAGELEVEL") & (d.arch == a) & (d.test == "in_source")].ba
    pt = d[(d.protocol == "type_figshare_raw") & (d.arch == a) & (d.test == "in_source")].ba
    rows.append(f"| {A[a]} | {ms(im)} | {ms(pt)} | {im.mean()-pt.mean():+.3f} |")
out["T_split"] = "\n".join(rows)
x = pd.read_csv(F + "addonC/external_eval_full_models.csv")
rows = ["| Training protocol | Representation | External set | BA (mean ± SD, 15 runs) | Macro-F1 | Bootstrap 95% CI (median run) |", "|---|---|---|---|---|---|"]
pl = {"bin_br35h": "Br35H, tumour vs none", "bin_sartaj": "SARTAJ, tumour vs none", "type_figshare": "Figshare, tumour type", "type_sartaj": "SARTAJ, tumour type", "four_nickparvar": "Nickparvar, 4-class"}
for (p, t), g in x.groupby(['protocol', 'test']):
    base = [k for k in pl if p.startswith(k)][0]
    rep = "harmonised" if p.endswith("harm") else ("raw, cluster split" if "CLUSTER" in p else ("raw, official split" if "OFFICIAL" in p else "raw"))
    med = g.iloc[(g.ba - g.ba.median()).abs().argsort()[:1]]
    rows.append(f"| {pl[base]} | {rep} | {'BDNeuro-MRI' if 'bdneuro' in t else 'PMRAM'} | {ms(g.ba)} | {g.macro_f1.mean():.3f} | {med.ba_ci_lo.iloc[0]:.3f}–{med.ba_ci_hi.iloc[0]:.3f} |")
out["T_ext"] = "\n".join(rows)
f = pd.read_csv(F + "addonC/foundation_model_probe.csv")
tn = {"in_source": "in-source", "official_test_clean": "duplicate-free test", "external_bdneuro": "BDNeuro-MRI", "external_pmram": "PMRAM"}
rows = ["| Training protocol | Test | BA (mean ± SD, 5 seeds) | Macro-F1 | AUROC |", "|---|---|---|---|---|"]
for (p, t), g in f.groupby(['protocol', 'test'], sort=False):
    base = [k for k in pl if p.startswith(k)][0]
    tt = tn.get(t, t.replace("cross_", "new source: ").replace("sartaj", "SARTAJ").replace("br35h", "Br35H").replace("figshare", "Figshare"))
    sp = " (official split)" if "OFFICIAL" in p else (" (cluster split)" if "CLUSTER" in p else "")
    au = f"{g.auroc.mean():.3f}" if g.auroc.notna().any() else "–"
    rows.append(f"| {pl[base]}{sp} | {tt} | {ms(g.ba)} | {g.macro_f1.mean():.3f} | {au} |")
out["T_dino"] = "\n".join(rows)
s = pd.read_csv(F + "addon/ccit_summary.csv")
tl = {"T1": "T1", "T2": "T2", "T3": "T3", "T4_in": "T4 (in-source)", "T4_cross": "T4 (new source)"}
pn = {"bin_br35h_raw": "Br35H (binary)", "bin_sartaj_raw": "SARTAJ (binary)", "type_figshare_raw": "Figshare (type)", "type_sartaj_raw": "SARTAJ (type)"}
rows = ["| Test | Protocol | Architecture | ΔBA (95% CI) | d_z | Flip rate | p (t) | p_Holm (t) | p_Holm (perm.) |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in s.sort_values(['test', 'protocol', 'arch']).iterrows():
    rows.append(f"| {tl[r.test]} | {pn[r.protocol]} | {A[r.arch]} | {r.mean_delta_ba:+.3f} ({r.ci_lo:+.3f}, {r.ci_hi:+.3f}) | {r.d_z:.2f} | {r.mean_flip_rate:.3f} | {r.t_p:.3f} | {r.t_p_holm:.2f} | {r.perm_p_holm:.2f} |".replace("nan", "–"))
out["T_ccit"] = "\n".join(rows)
at = pd.read_csv(F + "attribution_per_image.csv")
rows = ["| Architecture | Method | Set | Brain share | Background share | Background area | Background enrichment | n images |", "|---|---|---|---|---|---|---|---|"]
b = at[at.regions == "brain_vs_background"].copy(); b['setg'] = np.where(b.set == 'in_source', 'in-source', 'new source')
for (a, me, sg), g in b.groupby(['arch', 'method', 'setg']):
    rows.append(f"| {A[a]} | {'Grad-CAM' if me == 'gradcam' else 'Integrated Gradients'} | {sg} | {g.share_brain.mean():.3f} | {g.share_background.mean():.3f} | {g.area_background.mean():.3f} | {(g.share_background/g.area_background.clip(lower=1e-6)).median():.2f} | {g.img_id.nunique()} |")
out["T_attr"] = "\n".join(rows)
te = pd.read_csv(F + "tumour_enrichment.csv")
rows = ["| Architecture | Method | Tumour share | Tumour area | Enrichment (ratio of means) | Median per-image enrichment | Images with enrichment > 1 | Background share |", "|---|---|---|---|---|---|---|---|"]
for _, r in te.iterrows():
    rows.append(f"| {A[r.arch]} | {'Grad-CAM' if r.method == 'gradcam' else 'Integrated Gradients'} | {r.tumour_share:.3f} | {r.tumour_area:.3f} | {r.enrich_ratio_of_means:.2f} | {r.enrich_median_img:.2f} | {r.frac_img_enrich_gt1*100:.0f}% | {r.bg_share:.3f} |")
out["T_tumour"] = "\n".join(rows)
ss = pd.read_csv(F + "addonC/sourceshield_per_seed.csv")
vn = {"ERM_raw": "Standard training (ERM)", "ERM_harm": "Harmonisation", "mask_only": "Brain mask", "DRO_only": "Group DRO", "SourceShield": "SourceShield"}
rows = ["| Variant | Architecture | Same-source BA | BDNeuro-MRI BA | PMRAM BA | ECE (BDNeuro-MRI) |", "|---|---|---|---|---|---|"]
for v in vn:
    for a in ["resnet18", "efficientnet_b0"]:
        g = ss[(ss.model == v) & (ss.arch == a)]
        f_ = lambda t: ms(g[g.test == t].ba)
        rows.append(f"| {vn[v]} | {A[a]} | {f_('in_source')} | {f_('external_bdneuro')} | {f_('external_pmram')} | {g[g.test == 'external_bdneuro'].ece.mean():.3f} |")
out["T_rem_abs"] = "\n".join(rows)
L = pd.read_csv(F + "P1_literature_inflation_audit_pilot.csv")
rows = ["| # | Title (abridged) | Dataset family | Split reported | Duplicates / leakage | External test | Reported accuracy (%) | Classes |", "|---|---|---|---|---|---|---|---|"]
for i, r in L.iterrows():
    t = str(r.title); t = (t[:70] + "…") if len(t) > 70 else t
    ext = f"{r.external_test}{(' (' + str(r.external_set) + ')') if str(r.external_set) != 'nan' else ''}"
    rows.append(f"| {i+1} | {t} | {r.dataset_family} | {r.split_reported} | {r.dedup_or_leakage_mentioned} | {ext} | {'' if str(r.reported_acc) == 'nan' else r.reported_acc} | {r.n_classes} |")
out["T_lit"] = "\n".join(rows)
for k, v in out.items(): open(k + ".md", "w").write(v + "\n")
print("written:", ", ".join(out))
