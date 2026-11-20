"""组装 final 面板 (D/E/F 新 300 靶) 的 override 表, 沿用 v10 配方.

优先级: H1 实测 (Training12 + Val4 + HMGXB4) > orion+gwps unit-mix (w=0.4)
> orion-only > gwps-only > U9 嵌入 > 零行. 外部行按组把 filt-范数中位对齐到 H1 raw 锚.
H1 raw 行用与 build_lfc_override_v3 相同的估计量 (mean-CPM log2 ratio, base<1CPM 置零).
"""

import csv
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from linear_baseline import EPS, mean_cpm  # noqa: E402

OUT = "outputs/lfc_override_2026_final.npz"
W_ORION = 0.4
MIN_GWPS_CELLS = 10


def mean_cpm_idx(a, idx, chunk=4000):
    acc = np.zeros(a.n_vars, dtype=np.float64)
    n = 0
    for s in range(0, len(idx), chunk):
        part = a[idx[s:s + chunk]].to_memory()
        acc += mean_cpm(part).astype(np.float64) * len(part)
        n += len(part)
    return acc / max(n, 1)


def make_fill(var_names, gpos):
    return np.array([gpos.get(str(g), -1) for g in var_names])


def align_row(raw, fill, base_fit, n_genes, floor=1.0):
    row = np.zeros(n_genes, dtype=np.float32)
    ok = fill >= 0
    row[fill[ok]] = raw[ok]
    row[base_fit < floor] = 0.0
    return np.clip(np.nan_to_num(row), -4, 4)


def main():
    v10 = np.load("outputs/lfc_override_2026_v10.npz", allow_pickle=True)
    genes = v10["genes"].astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    filt = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)["filt"]

    panel = [r["target_gene"].strip() for r in
             csv.DictReader(open("data/final/pert_counts.csv"))]
    assert len(panel) == 300 and len(set(panel)) == 300

    rows = {}

    # ---- H1 raw injections ----
    train_names = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
    train12 = [g for g in panel if g in train_names]
    holdout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    hold4 = [g for g in panel if g in holdout]
    val_names = set(r[0].strip() for r in
                    list(csv.reader(open("data/pert_counts_Validation_2025.csv")))[1:])
    val_extra = [g for g in panel if g in val_names and g not in train_names and g not in holdout]
    print(f"H1 raw: train {len(train12)} {train12}")
    print(f"        holdout {len(hold4)} {hold4}")
    print(f"        val-train-split {len(val_extra)} {val_extra}")

    ctrl = ad.read_h5ad("data/internal/train_controls.h5ad", backed="r")
    base_tr = mean_cpm_idx(ctrl, np.arange(len(ctrl)))
    fill_tr = make_fill(ctrl.var_names, gpos)
    base_fit_tr = np.zeros(n_genes)
    ok = fill_tr >= 0
    base_fit_tr[fill_tr[ok]] = base_tr[fill_tr[ok]]
    ctrl.file.close()

    a_tr = ad.read_h5ad("data/adata_Training_2025.h5ad", backed="r")
    labels_tr = a_tr.obs["target_gene"].astype(str).to_numpy()
    fill_a_tr = make_fill(a_tr.var_names, gpos)
    rng = np.random.default_rng(0)
    for g in train12:
        idx = np.where(labels_tr == g)[0]
        if len(idx) > 400:
            idx = rng.choice(idx, 400, replace=False)
        assert len(idx) > 0, g
        m = mean_cpm(a_tr[np.sort(idx)].to_memory())
        raw = np.log2((m + EPS) / (base_tr + EPS))
        rows[g] = align_row(raw, fill_a_tr, base_fit_tr, n_genes)
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f}")
    a_tr.file.close()

    a_val = ad.read_h5ad("data/internal/real_heldout.h5ad", backed="r")
    labels_val = a_val.obs["target_gene"].astype(str).to_numpy()
    fill_a_val = make_fill(a_val.var_names, gpos)
    base_val = mean_cpm_idx(a_val, np.where(labels_val == "non-targeting")[0])
    base_fit_val = np.zeros(n_genes)
    ok = fill_a_val >= 0
    base_fit_val[fill_a_val[ok]] = base_val[fill_a_val[ok]]
    for g in hold4:
        idx = np.where(labels_val == g)[0][:400]
        assert len(idx) > 0, g
        m = mean_cpm(a_val[idx].to_memory())
        raw = np.log2((m + EPS) / (base_val + EPS))
        rows[g] = align_row(raw, fill_a_val, base_fit_val, n_genes)
        print(f"  {g}: {len(idx)} cells, norm {np.linalg.norm(rows[g]):.1f}")
    a_val.file.close()

    if val_extra:
        a_tp = ad.read_h5ad("data/internal/train_perturbed.h5ad", backed="r")
        labels_tp = a_tp.obs["target_gene"].astype(str).to_numpy()
        fill_a_tp = make_fill(a_tp.var_names, gpos)
        for g in val_extra:
            idx = np.where(labels_tp == g)[0][:400]
            assert len(idx) > 0, g
            m = mean_cpm(a_tp[idx].to_memory())
            raw = np.log2((m + EPS) / (base_val + EPS))
            rows[g] = align_row(raw, fill_a_tp, base_fit_val, n_genes)
            print(f"  {g}: {len(idx)} cells (val train split), norm "
                  f"{np.linalg.norm(rows[g]):.1f}")
        a_tp.file.close()

    raw_set = set(rows)
    n_raw = np.median([np.linalg.norm(v[filt]) for v in rows.values()])
    print(f"raw anchor median filt-norm: {n_raw:.2f} ({len(rows)} rows)")

    # sanity: shared target with v3 must match
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    v3_tab = {str(g): v3["lfc"][i] for i, g in enumerate(v3["names"].astype(str))}
    for g in sorted(raw_set & set(v3_tab)):
        d = np.abs(rows[g] - v3_tab[g]).max()
        print(f"  sanity {g}: max|new-v3| {d:.2e}")

    # ---- external rows ----
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gpos:
            ocols.append(oc)
            opos.append(gpos[g])
    ocols = np.array(ocols)
    opos = np.array(opos)
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    L = np.load("data/external/orion_lfc_true.npy", mmap_mode="r")
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    rowmap = {}
    for r in range(L.shape[0]):
        rowmap.setdefault(ctx_name[int(ctx_ids[r])], {})[int(pert_ids[r])] = r

    def orion_natural(t):
        pid = pert_to_id.get(t)
        if pid is None:
            return None
        vs = []
        for cl in ("hct116", "hek293t"):
            r = rowmap[cl].get(pid)
            if r is None:
                continue
            v = np.asarray(L[r, ocols], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if np.abs(v).sum() > 0:
                vs.append(v)
        if not vs:
            return None
        full = np.zeros(n_genes)
        full[opos] = np.mean(vs, 0)
        full[~filt] = 0.0
        return full if np.abs(full).sum() > 0 else None

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gcells = g["n_cells"].astype(int)
    gmap = {nm: i for i, nm in enumerate(gnames)}

    def gwps_row(t):
        j = gmap.get(t)
        if j is None or gcells[j] < MIN_GWPS_CELLS:
            return None, (int(gcells[j]) if j is not None else None)
        v = np.nan_to_num(grows[j], nan=0.0, posinf=0.0, neginf=0.0)
        v[~filt] = 0.0
        return (v if np.abs(v).sum() > 0 else None), int(gcells[j])

    pending = [t for t in panel if t not in raw_set]
    orion, gwps, low_cells = {}, {}, []
    for t in pending:
        ov = orion_natural(t)
        if ov is not None:
            orion[t] = ov
        gv, nc = gwps_row(t)
        if gv is not None:
            gwps[t] = gv
        elif nc is not None and nc < MIN_GWPS_CELLS:
            low_cells.append((t, nc))
    print(f"external: orion {len(orion)}, gwps {len(gwps)}, gwps low-cells {low_cells}")

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    mixed = {t: np.where(filt, W_ORION * unit(orion[t]) + (1 - W_ORION) * unit(gwps[t]), 0.0)
             for t in orion if t in gwps}
    orion_only = {t: v for t, v in orion.items() if t not in gwps}
    gwps_only = {t: v for t, v in gwps.items() if t not in orion}
    print(f"groups: mixed {len(mixed)}, orion-only {len(orion_only)}, "
          f"gwps-only {len(gwps_only)}")

    def scaled(group, name):
        if not group:
            return
        nats = np.array([np.linalg.norm(v[filt]) for v in group.values()])
        c = n_raw / np.median(nats)
        print(f"  {name}: natural median {np.median(nats):.3f} -> c {c:.4g}")
        for t, v in group.items():
            rows[t] = np.clip(c * v, -4, 4).astype(np.float32)

    scaled(mixed, "unit-mix")
    scaled(orion_only, "orion-only")
    scaled(gwps_only, "gwps-only")

    # ---- embedding fallback (U9) ----
    ens = np.load("outputs/lfc_add_emb_ens.npz", allow_pickle=True)
    ens_names = [str(x) for x in ens["names"]]
    ens_map = {t: i for i, t in enumerate(ens_names)}
    ens_rows = {t: np.where(filt, ens["lfc"][ens_map[t]], 0.0).astype(np.float64)
                for t in panel if t not in rows and t in ens_map}
    if ens_rows:
        nats = np.array([np.linalg.norm(v[filt]) for v in ens_rows.values()])
        c = n_raw / np.median(nats)
        print(f"  U9: {len(ens_rows)} rows, natural median {np.median(nats):.3f} -> c {c:.2f}")
        for t, v in ens_rows.items():
            rows[t] = np.clip(c * v, -4, 4).astype(np.float32)

    zero = [t for t in panel if t not in rows]
    for t in zero:
        rows[t] = np.zeros(n_genes, dtype=np.float32)
    print(f"zero rows {len(zero)} {zero}")

    tab = np.stack([rows[t] for t in panel]).astype(np.float32)
    assert tab.shape == (300, n_genes) and np.isfinite(tab).all()
    print(f"final table {tab.shape}, max|.| {np.abs(tab).max():.2f}")
    print(f"row norm median {np.median(np.linalg.norm(tab[:, filt], axis=1)):.2f}")
    np.savez(OUT, names=np.array(panel), genes=genes, lfc=tab)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
