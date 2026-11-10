"""Generate the computed tables (snippets T_*.md) from the saved result files. Run from ms2/."""
import pandas as pd, numpy as np
import os
F = os.environ.get("P1_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")) + "/"; D = F + "run_tables/"
A = {"resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-T"}
SRC = {"figshare": "Figshare", "sartaj": "SARTAJ", "br35h": "Br35H", "nickparvar": "Nickparvar", "brisc": "BRISC"}
out = {}

# Table 7 containment matrix (rows container A, cols B)
c = pd.read_csv(D + "containment_matrix.csv")
order = ["figshare", "sartaj", "br35h", "nickparvar", "brisc"]
rows = ["| Contains → / is contained ↓ | " + " | ".join(SRC[b] for b in order) + " |", "|---|" + "---|" * 5]
for a in order:
    cells = []
    for b in order:
        if a == b: cells.append("—"); continue
        r = c[(c.container_A == a) & (c.corpus_B == b)].iloc[0]
        cells.append(f"{r.pct_B_in_A:.1f} ({r.pct_B_exact_in_A:.1f})")
    rows.append(f"| {SRC[a]} | " + " | ".join(cells) + " |")
out["T_contain"] = "\n".join(rows)

# Table 8 NCC sweep
s = pd.read_csv(D + "ncc_threshold_sweep.csv")
rows = ["| Threshold τ | Near-duplicate clusters | Images in multi-image clusters | Share of all images (%) |", "|---|---|---|---|"]
for r in s.itertuples():
    rows.append(f"| {r.threshold:g}{' (used)' if r.threshold == 0.98 else ''} | {r.n_clusters:,} | {r.images_in_multi_clusters:,} | {100*r.images_in_multi_clusters/22528:.1f} |")
out["T_sweep"] = "\n".join(rows)

# Table 9 replication
rp = pd.read_csv(F + "replication_per_seed.csv")
nm = {"mlhnet_style": "Kaprekar-style", "khan_cnn": "Khan-style"}
tl = [("official_test", "Official test"), ("official_test_clean", "Duplicate-free test"), ("cluster_test", "Cluster-disjoint split"), ("cross_figshare", "New source (SARTAJ → Figshare)")]
rows = ["| Model | " + " | ".join(t[1] for t in tl) + " |", "|---|---|---|---|---|"]
for k in ["mlhnet_style", "khan_cnn"]:
    cells = []
    for t, _ in tl:
        v = rp[(rp.arch == k) & (rp.test == t)].ba
        cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
    rows.append(f"| {nm[k]} | " + " | ".join(cells) + " |")
out["T_repl"] = "\n".join(rows)

# Table 12 H1/H3 tests (from RESULTS_AUDIT_FULL) + check against per-seed data
st = pd.read_csv(D + "stats_tests_from_audit.csv")
d = pd.read_csv(F + "per_seed_metrics_FULL.csv")
dirs = {"bin_br35h": "Br35H → SARTAJ", "bin_sartaj": "SARTAJ → Br35H", "type_figshare": "Figshare → SARTAJ", "type_sartaj": "SARTAJ → Figshare"}
rows = ["| Direction | Architecture | Drop, raw (p_Holm; d_z) | Drop, harmonised (p_Holm; d_z) | Harmonised − raw on new source (p_Holm; d_z) |", "|---|---|---|---|---|"]
chk = []
for p, lab in dirs.items():
    for a in ["resnet18", "efficientnet_b0", "convnext_tiny"]:
        f = lambda h, comp: st[(st.hypothesis == h) & (st.comparison == comp) & (st.arch == a)].iloc[0]
        r1, r2, r3 = f("H1", p + "_raw"), f("H1", p + "_harm"), f("H3", p)
        fmtp = lambda x: "<0.001" if x < 0.001 else (f"{x:.3f}" if x < 0.1 else f"{x:.2f}")
        rows.append(f"| {lab} | {A[a]} | {r1['diff']:+.3f} ({fmtp(r1.p_holm)}; {r1.d_z:.2f}) | {r2['diff']:+.3f} ({fmtp(r2.p_holm)}; {r2.d_z:.2f}) | {r3['diff']:+.3f} ({fmtp(r3.p_holm)}; {r3.d_z:.2f}) |")
        # recompute drop from per-seed rows
        for rep, rr in [("raw", r1), ("harm", r2)]:
            g = d[(d.protocol == f"{p}_{rep}") & (d.arch == a)]
            ins = g[g.test == "in_source"].set_index("seed").ba; cr = g[g.test.str.startswith("cross")].set_index("seed").ba
            chk.append((f"H1 {p}_{rep} {a}", rr["diff"], round((ins - cr).mean(), 3)))
        gr = d[(d.protocol == f"{p}_raw") & (d.arch == a) & d.test.str.startswith("cross")].set_index("seed").ba
        gh = d[(d.protocol == f"{p}_harm") & (d.arch == a) & d.test.str.startswith("cross")].set_index("seed").ba
        chk.append((f"H3 {p} {a}", r3["diff"], round((gh - gr).mean(), 3)))
out["T_tests"] = "\n".join(rows)
pd.DataFrame(chk, columns=["item", "audit", "recomputed"]).to_csv("check_H_tests.csv", index=False)

# Table 16 probes
pr = pd.read_csv(D + "probe_results.csv")
pn = {"background_raw": "Background, raw", "background_harm": "Background, harmonised", "format_raw": "File format"}
tn = {("bin", "br35h"): "Br35H, tumour vs none", ("bin", "sartaj"): "SARTAJ, tumour vs none", ("type", "figshare"): "Figshare, tumour type", ("type", "sartaj"): "SARTAJ, tumour type"}
rows = ["| Feature set | Training source and task | Chance BA | In-source BA (grouped CV, mean ± SD) | New-source BA | n (in / new) |", "|---|---|---|---|---|---|"]
for p in pn:
    for (task, src), lab in tn.items():
        g = pr[(pr.probe == p) & (pr.task == task) & (pr.train_src == src)]
        i = g[g.test == "in_source_groupedCV"].iloc[0]; x = g[g.test != "in_source_groupedCV"].iloc[0]
        rows.append(f"| {pn[p]} | {lab} | {i.chance:.2f} | {i.ba_mean:.3f} ± {i.ba_sd:.3f} | {x.ba_mean:.3f} | {int(i.n):,} / {int(x.n):,} |")
out["T_probe"] = "\n".join(rows)

# Table 17 CCIT summary per test
cc = pd.read_csv(F + "addon/ccit_summary.csv")
tl = {"T1": "T1 background transplant (new-source images)", "T2": "T2 reverse transplant (in-source images)", "T3": "T3 re-encoding (in-source files)",
      "T4_in": "T4 remove extra-cranial pixels (in-source)", "T4_cross": "T4 remove extra-cranial pixels (new source)"}
rows = ["| Test | Conditions | ΔBA range | Largest \\|ΔBA\\| | Mean flip rate | Max flip rate | Smallest p_Holm (t) | Smallest p_Holm (permutation) | Meets criterion |", "|---|---|---|---|---|---|---|---|---|"]
for t, lab in tl.items():
    g = cc[cc.test == t]
    rows.append(f"| {lab} | {len(g)} | {g.mean_delta_ba.min():+.3f} to {g.mean_delta_ba.max():+.3f} | {g.mean_delta_ba.abs().max():.3f} | {g.mean_flip_rate.mean():.3f} | {g.mean_flip_rate.max():.3f} | {g.t_p_holm.min():.2f} | {g.perm_p_holm.min():.2f} | 0 of {len(g)} |")
rows.append(f"| All | {len(cc)} | {cc.mean_delta_ba.min():+.3f} to {cc.mean_delta_ba.max():+.3f} | {cc.mean_delta_ba.abs().max():.3f} | {cc.mean_flip_rate.mean():.3f} | {cc.mean_flip_rate.max():.3f} | {cc.t_p_holm.min():.2f} | {cc.perm_p_holm.min():.2f} | 0 of {len(cc)} |")
out["T_ccit_sum"] = "\n".join(rows)

# Table 20 remedy gains
ss = pd.read_csv(F + "addonC/sourceshield_stats.csv")
vn = {"ERM_harm": "Harmonisation", "mask_only": "Brain mask", "DRO_only": "Group DRO", "SourceShield": "SourceShield"}
rows = ["| Remedy | Architecture | Same source | BDNeuro-MRI | PMRAM |", "|---|---|---|---|---|"]
for v in vn:
    for a in ["resnet18", "efficientnet_b0"]:
        cells = []
        for t in ["in_source", "external_bdneuro", "external_pmram"]:
            r = ss[(ss.variant == v) & (ss.arch == a) & (ss.test == t)].iloc[0]
            meets = r.mean_diff_ba >= 0.05 and r.ci_lo > 0
            val = f"{r.mean_diff_ba:+.3f} ({r.ci_lo:+.3f}, {r.ci_hi:+.3f}); p_Holm {r.t_p_holm:.2f}"
            cells.append(f"**{val}**" if meets else val)
        rows.append(f"| {vn[v]} | {A[a]} | " + " | ".join(cells) + " |")
out["T_rem_gain"] = "\n".join(rows)

# Table 22 literature summary
L = pd.read_csv(F + "P1_literature_inflation_audit_pilot.csv")
out["lit_cols"] = str(L.columns.tolist())
for k, v in out.items():
    if k.startswith("T_"): open(k + ".md", "w").write(v + "\n")
print(pd.read_csv("check_H_tests.csv").assign(ok=lambda x: (x.audit - x.recomputed).abs() <= 0.0015).ok.value_counts())
print(out["lit_cols"])
for c_ in ["dataset_family", "split_reported", "dedup_or_leakage_mentioned", "external_test"]:
    print(L[c_].value_counts(dropna=False).to_dict())
print(L.reported_acc.describe())
