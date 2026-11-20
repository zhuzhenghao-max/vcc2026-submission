# Override-table build chain

All scripts are run from the repository root. They expect a working tree with
`data/`, `outputs/`, and `src/` laid out as described in `data/README.md`.
Inline comments in the copied scripts are in Chinese, as originally written.

The full chain that produced the submitted table:

```
v1   build_lfc_override.py        scVI-decoded H1 rows
v2   build_lfc_override_v2.py     raw H1 responses replace decoded rows
v3   build_lfc_override_v3.py     genes below 1 CPM set to zero
v4   build_ov4.py                 embedding fallback rows
v5   build_ov5.py                 more embedding donors
v6   build_ov6.py                 Orion rows
v7   build_ov7.py                 GWPS rows
v8   build_ov8.py                 combined external rows
v9   build_ov9.py                 FDR-aware variant (exploratory)
v10  build_ov10.py                measured rows replace eight decoded rows
     build_ov_final.py            final 300-target panel, v10 recipe
     build_ov11.py                gated PerturbAtlas rows
```

Only the last three builders and the v10 step are included here. The earlier
ones live in the companion benchmark repository
(<https://github.com/zhuzhenghao-max/mixing-vs-selection>) under `scripts/`.
The intermediate tables themselves are not redistributed.

## Included scripts

### `build_ov10.py`

Inputs:

```
outputs/lfc_override_2026_v8.npz
outputs/lfc_override_2026_v7.npz
outputs/lfc_override_2026_v3.npz
outputs/scvi_map_cache.npz            # filt mask
outputs/gwps_rows.npz
data/internal/holdout_genes.csv
data/vccp_de_log2fc.parquet
data/external/orion_meta.json
data/external/orion_lfc_true.npy
data/external/orion_pert_ids.npy
data/external/orion_ctx_ids.npy
```

Output: `outputs/lfc_override_2026_v10.npz`.

### `build_ov_final.py`

Inputs:

```
outputs/lfc_override_2026_v10.npz
outputs/lfc_override_2026_v3.npz     # sanity check only
outputs/scvi_map_cache.npz           # filt mask
outputs/gwps_rows.npz
outputs/lfc_add_emb_ens.npz
data/final/pert_counts.csv           # 300-target final panel
data/adata_Training_2025.h5ad
data/internal/train_controls.h5ad
data/internal/train_perturbed.h5ad
data/internal/real_heldout.h5ad
data/internal/holdout_genes.csv
data/pert_counts_Validation_2025.csv
data/vccp_de_log2fc.parquet
data/external/orion_meta.json
data/external/orion_lfc_true.npy
data/external/orion_pert_ids.npy
data/external/orion_ctx_ids.npy
```

Output: `outputs/lfc_override_2026_final.npz`.

### `build_ov11.py`

Inputs:

```
outputs/lfc_override_2026_final.npz
outputs/scvi_map_cache.npz
outputs/gwps_rows.npz
outputs/lfc_add_emb_ens.npz
outputs/perturbatlas_panel_comps.json
outputs/perturbatlas_cand_totals.json
outputs/perturbatlas_deg/*.xlsx
data/external/hgnc_complete_set.txt
data/external/orion_meta.json
data/external/orion_pert_ids.npy
data/external/orion_ctx_ids.npy
data/internal/holdout_genes.csv
data/pert_counts_Validation_2025.csv
data/vccp_de_log2fc.parquet
```

Outputs: `outputs/lfc_override_2026_v11.npz` and
`outputs/v11_pa_changes.json`.

The PerturbAtlas tables can be fetched with
`scripts/perturbatlas_dl_panel.py` from the companion repository once the
panel comparison list is available.
