# Present but Unused: Dataset Shortcuts Do Not Explain Why Brain-Tumour MRI Classifiers Fail on New Sources

Code and result files for the study of near-duplicate leakage, source shift and shortcut use in public brain-tumour MRI classifiers. The package reproduces every table and figure of the manuscript and its supplementary file.

Repository: https://github.com/IsaacJournals/brain-mri-source-shift-audit

## Open the notebooks in Colab

| Notebook | Launch |
|---|---|
| 01 main pipeline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaacJournals/brain-mri-source-shift-audit/blob/main/notebooks/01_P1_main_pipeline.ipynb) |
| 02 base-paper replication | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaacJournals/brain-mri-source-shift-audit/blob/main/notebooks/02_P1_base_paper_replication.ipynb) |
| 03 causal cue tests and attribution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaacJournals/brain-mri-source-shift-audit/blob/main/notebooks/03_P1_causal_cue_tests_attribution.ipynb) |
| 04 external tests, remedies, DINOv2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaacJournals/brain-mri-source-shift-audit/blob/main/notebooks/04_P1_external_remedies_dinov2.ipynb) |
| 05 tables and figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IsaacJournals/brain-mri-source-shift-audit/blob/main/notebooks/05_P1_tables_and_figures.ipynb) |

## Contents

| Folder / file | What it contains |
|---|---|
| `notebooks/01_P1_main_pipeline.ipynb` | Main analysis (Algorithms 1–4): data download, near-duplicate lineage map, containment matrix, group-disjoint splits, 165 training runs (3 architectures × 11 protocols × 5 seeds), evaluation, shortcut probes, statistics |
| `notebooks/02_P1_base_paper_replication.ipynb` | Khan-style and Kaprekar-style re-implementations on official, duplicate-free, cluster-disjoint and new-source tests |
| `notebooks/03_P1_causal_cue_tests_attribution.ipynb` | Causal Cue Intervention Test (T1–T4) and Grad-CAM / Integrated Gradients attribution (Algorithms 5–6) |
| `notebooks/04_P1_external_remedies_dinov2.ipynb` | External tests on non-recycled BDNeuro-MRI and PMRAM images, remedy comparison, frozen DINOv2 reference (Algorithm 7) |
| `notebooks/05_P1_tables_and_figures.ipynb` | Rebuilds all manuscript tables and figures from `results/` (no GPU) |
| `src/` | The Python modules run by the notebooks (`P1_pipeline.py`, `P1_replication.py`, `P1_addon.py`, `P1_addonC.py`) |
| `analysis/` | Table and figure scripts used for the manuscript |
| `results/` | Result files of the confirmatory run (CSV) from which every reported number is computed |
| `requirements.txt` | Package versions of the confirmatory run |

## Running the analysis in Google Colab

1. Open a notebook with its Colab button above. Notebooks 03 and 04 clone this repository automatically; to work from a Drive copy instead, upload the repository folder to `MyDrive/P1_code_release` and set `GITHUB_URL = ''`.
2. In Colab, add a secret named `KAGGLE_API_TOKEN` (Kaggle → Settings → API → Create new token; key icon in the left bar; switch notebook access on). The Figshare, BDNeuro-MRI and PMRAM files are downloaded without credentials.
3. Run the notebooks in order with **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**, and allow Google Drive access:

| Order | Notebook | GPU time (T4) | Needs |
|---|---|---|---|
| 1 | 01 main pipeline | about 7 h | Kaggle secret |
| 2 | 02 replication | about 3 h | output of 01 |
| 3 | 03 causal cue tests and attribution | about 1.1 h | checkpoints of 01 |
| 4 | 04 external tests, remedies, DINOv2 | about 4–5 h | checkpoints of 01 |
| 5 | 05 tables and figures | none (2 min) | `results/` |

Run one GPU notebook at a time. All outputs are written to `MyDrive/MI10_P1/`; every training and evaluation loop saves its progress, so after a disconnect **Run all** resumes where it stopped. Setting `MODE = "SMOKE_TEST"` in the configuration cell of notebook 01 runs a 10-minute plumbing check; its numbers are not valid results.

## Data

| Source | Access in the code | Licence / terms |
|---|---|---|
| Figshare brain tumour dataset (Cheng), doi:10.6084/m9.figshare.1512427 | direct download | CC BY 4.0 |
| SARTAJ, Kaggle `sartajbhuvaji/brain-tumor-classification-mri` | Kaggle API | Kaggle dataset terms |
| Br35H, Kaggle `ahmedhamada0/brain-tumor-detection` | Kaggle API | Kaggle dataset terms |
| Nickparvar, Kaggle `masoudnickparvar/brain-tumor-mri-dataset` | Kaggle API | Kaggle dataset terms |
| BRISC 2025, Kaggle `briscdataset/brisc2025` | Kaggle API | CC BY 4.0 |
| BDNeuro-MRI v8, doi:10.17632/zwr4ntf94j.8 | Mendeley Data API | as stated on the Mendeley Data page |
| PMRAM v1, doi:10.17632/m7w55sw88b.1 (raw images only) | Mendeley Data API | as stated on the Mendeley Data page |

No images are redistributed in this package. `results/manifest_with_clusters.csv` lists every image by source, file path, label, official split, MD5 checksum, near-duplicate cluster and group label, so that the duplicate-free splits can be rebuilt from the original downloads.

## Where each manuscript result comes from

| Manuscript item | Result file(s) in `results/` | Script |
|---|---|---|
| Table 4 (images per source and class) | `manifest_with_clusters.csv` | `make_tables_main.py` |
| Tables 7, 8 (containment, threshold sweep) | `run_tables/containment_matrix.csv`, `run_tables/ncc_threshold_sweep.csv` | `make_tables.py` |
| Table 9 (replication) | `replication_per_seed.csv` | `make_tables.py` |
| Tables 10, 11, 13 (per-class, cross-source, split level) | `per_seed_metrics_FULL.csv` | `make_tables_main.py` |
| Table 12 (paired tests H1, H3) | `run_tables/stats_tests_from_audit.csv`, checked against `per_seed_metrics_FULL.csv` | `make_tables.py` |
| Tables 14, 15 (external, DINOv2) | `addonC/external_eval_full_models.csv`, `addonC/foundation_model_probe.csv` | `make_tables_main.py` |
| Table 16 (probes) | `run_tables/probe_results.csv` | `make_tables.py` |
| Table 17, Supplementary Table S1 (CCIT) | `addon/ccit_summary.csv`, `addon/ccit_per_seed.csv` | `make_tables.py`, `make_tables_main.py` |
| Tables 18, 19 (attribution) | `attribution_per_image.csv`, `tumour_enrichment.csv` | `make_tables_main.py` |
| Tables 20, 21 (remedies) | `addonC/sourceshield_stats.csv`, `addonC/sourceshield_per_seed.csv` | `make_tables.py`, `make_tables_main.py` |
| Table 22, Supplementary Table S2 (literature audit) | `P1_literature_inflation_audit_pilot.csv` | `make_tables_main.py` |
| Fig. 1 | — (diagram) | `make_fig1_study_design.py` |
| Figs. 2–5 | — (diagrams) | `make_diagrams.py` |
| Figs. 6–12 | files above | `make_figures.py` (output names `Fig2_containment` … `Fig8_literature` correspond to manuscript Figs. 6–12) |

`results/clean_test_glioma_74_for_review.csv` lists the 74 glioma images of the duplicate-free Nickparvar test (Section 7.3) for independent clinical review.

## Reproducibility notes

- Seeds 0–4 are used for every configuration; group-disjoint splits are drawn per seed, and the pipeline stops if any group appears in two partitions.
- Mixed-precision training on a GPU makes repeated runs agree closely but not bit-for-bit.
- The full package list of the confirmatory run is written by notebook 01 to `MyDrive/MI10_P1/project_outputs/config/env_lock.txt`; the main versions are in `requirements.txt`.
- Each run writes `RESULTS_AUDIT.md`, a claim-to-evidence table with the gate decisions of the pre-declared hypotheses.

## Licence

Code: MIT licence (see `LICENSE`). Result files: CC BY 4.0. The image datasets remain under the terms of their providers.
