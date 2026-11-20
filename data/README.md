# Data layout

The scripts expect the following paths, relative to the repository root.
None of the data files are included in this repository.

```
data/adata_Training_2025.h5ad              H1 training split, 2025 challenge
data/vccp_de_log2fc.parquet                H1 log-FC summary
data/pert_counts_Validation_2025.csv       validation target list
data/internal/train_controls.h5ad          H1 non-targeting controls
data/internal/train_perturbed.h5ad         H1 training perturbations
data/internal/real_heldout.h5ad            H1 validation perturbations
data/internal/holdout_genes.csv            held-out target list
data/final/pert_counts.csv                 final 300-target panel
data/final/context_D.h5ad                  final context D controls
data/final/context_E.h5ad                  final context E controls
data/final/context_F.h5ad                  final context F controls
data/external/orion_meta.json              X-Atlas/Orion metadata
data/external/orion_lfc_true.npy           Orion log fold changes
data/external/orion_pert_ids.npy           Orion perturbation ids
data/external/orion_ctx_ids.npy            Orion context ids
data/external/hgnc_complete_set.txt        HGNC symbols
outputs/gwps_rows.npz                      Replogle GWPS response rows
outputs/scvi_map_cache.npz                 gene axis and filter mask
outputs/lfc_add_emb_ens.npz                embedding fallback rows
```

## Sources

| Data | Source | Notes |
|---|---|---|
| H1 training and validation splits | 2025 Virtual Cell Challenge | Released training data; no test truth used |
| Replogle genome-wide Perturb-seq | Replogle et al., *Cell* (2022) | K562 screen |
| X-Atlas/Orion | X-Atlas/Orion release | HCT116 and HEK293T |
| PerturbAtlas DEG tables | <https://perturbatlas.kratoss.site> | Downloaded per comparison |
| HGNC complete set | <https://www.genenames.org/download/archive/> | Symbol mapping |

The 2026 challenge reference contexts D/E/F were served through the
challenge CLI while the competition was open. They are no longer available
for download after closure, so the exact final submission cannot be
regenerated without them.
