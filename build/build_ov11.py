"""v11: final 表上把 PerturbAtlas 跨系行混入非 raw 靶.

规则 (H1 验证: PDS +0.063, reach +0.072, fid/jac 不变, n=24):
  - 每靶取 total>=5000 的 KD/KO 对比; 无则跳过
  - gate: cov>=5k 对比中至少一个 self_lfc <= -1 (敲低有效)
  - 聚合: 按 max(0,-self_lfc) 加权的 unit 行均值
  - 混合: 0.4*unit(PA) + 0.6*unit(existing), 组尺度对齐 raw17 锚
  - raw17 行、U9 行、零行一律不动
"""

import csv
import json

import numpy as np
import pandas as pd

W_PA = 0.4
MIN_COV = 5000
SELF_THR = -1.0
OUT = "outputs/lfc_override_2026_v11.npz"


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def load_hgnc_map():
    sym_of = {}
    with open("data/external/hgnc_complete_set.txt", encoding="utf-8") as f:
        for x in csv.DictReader(f, delimiter="\t"):
            e = x.get("ensembl_gene_id")
            if e and e not in sym_of:
                sym_of[e] = x["symbol"]
    return sym_of


def build_row(path, genes, sym_of, gidx, tgt):
    df = pd.read_excel(path)
    df["sym"] = df["gene"].map(sym_of)
    df = df[df["sym"].isin(gidx)]
    ax = df["sym"].map(gidx).to_numpy()
    v = np.zeros(len(genes))
    cnt = np.zeros(len(genes))
    np.add.at(v, ax, df["log2FoldChange"].to_numpy())
    np.add.at(cnt, ax, 1.0)
    v = v / np.maximum(cnt, 1.0)
    ti = gidx.get(tgt)
    self_lfc = float(v[ti]) if ti is not None and cnt[ti] > 0 else np.nan
    return v, self_lfc, int((cnt > 0).sum())


def main():
    v10 = np.load("outputs/lfc_override_2026_final.npz", allow_pickle=True)
    panel = [str(x) for x in v10["names"]]
    genes = v10["genes"].astype(str)
    gidx = {g: i for i, g in enumerate(genes)}
    tab = v10["lfc"].copy()
    filt = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)["filt"]

    train_names = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
    train12 = [g for g in panel if g in train_names]
    holdout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    hold4 = [g for g in panel if g in holdout]
    val_names = set(r[0].strip() for r in
                    list(csv.reader(open("data/pert_counts_Validation_2025.csv")))[1:])
    val_extra = [g for g in panel if g in val_names and g not in train_names
                 and g not in holdout]
    raw_set = set(train12) | set(hold4) | set(val_extra)
    assert len(raw_set) == 17, sorted(raw_set)
    print(f"raw17 ({len(raw_set)}): {sorted(raw_set)}")

    # 无需改动的组: U9 与零行
    ens = np.load("outputs/lfc_add_emb_ens.npz", allow_pickle=True)
    ens_names = set(str(x) for x in ens["names"])
    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    gcells = g["n_cells"].astype(int)
    gmap = {t: i for i, t in enumerate(gnames)}
    gwps = set(t for t in panel if t in gmap and gcells[gmap[t]] >= 10
               and np.abs(g["rows"][gmap[t]]).sum() > 0)
    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    pert_to_id = meta["pert_to_id"]
    ctx_name = {v: k for k, v in meta["context_to_id"].items()}
    pert_ids = np.load("data/external/orion_pert_ids.npy")
    ctx_ids = np.load("data/external/orion_ctx_ids.npy")
    have = set(int(pert_ids[r]) for r in range(len(pert_ids))
               if ctx_name.get(int(ctx_ids[r])) in ("hct116", "hek293t"))
    orion = set(t for t in panel if t in pert_to_id and pert_to_id[t] in have)
    u9 = set(t for t in panel if t not in gwps and t not in orion
             and t not in raw_set and t in ens_names)
    print(f"u9 rows kept untouched: {sorted(u9)}")

    comps = json.load(open("outputs/perturbatlas_panel_comps.json", encoding="utf-8"))
    totals = {x["perturb_id"]: x["total"] for x in
              json.load(open("outputs/perturbatlas_cand_totals.json", encoding="utf-8"))}
    per = {}
    for x in comps:
        if x["condition"] not in ("KD", "KO"):
            continue
        pid = x["perturb_id"]
        if pid not in totals:
            continue
        per.setdefault(x["target"], []).append(dict(pid=pid, total=totals[pid],
                                                    cond=x["condition"],
                                                    tissue=x["tissue"]))
    sym_of = load_hgnc_map()

    changed, skipped = {}, []
    print("\n=== target decisions ===")
    for t in sorted(per):
        if t in raw_set:
            skipped.append((t, "raw"))
            continue
        if t in u9:
            skipped.append((t, "u9"))
            continue
        cov5 = [c for c in per[t] if c["total"] >= MIN_COV]
        if not cov5:
            skipped.append((t, "cov<5k"))
            continue
        rows, selfs, metas = [], [], []
        for c in cov5:
            path = f"outputs/perturbatlas_deg/{c['pid']}_degs.xlsx"
            try:
                v, self_lfc, n_ax = build_row(path, genes, sym_of, gidx, t)
            except Exception as e:
                print(f"  FAIL {t} {c['pid']}: {e}")
                continue
            rows.append(v)
            selfs.append(self_lfc)
            metas.append(c)
        fin = [x for x in selfs if np.isfinite(x)]
        mins = min(fin) if fin else np.inf
        if mins > SELF_THR:
            skipped.append((t, f"gate({mins:+.2f})"))
            print(f"  {t:9s} SKIP gate min_self {mins:+.2f} "
                  f"self={['nan' if not np.isfinite(x) else f'{x:+.2f}' for x in selfs]}")
            continue
        wts = np.array([max(0.0, -x) if np.isfinite(x) else 0.0 for x in selfs])
        assert wts.sum() > 0, t
        P = np.stack([np.where(filt, v, 0.0) for v in rows])
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        pa = (Pn * wts[:, None]).sum(0) / wts.sum()
        i = panel.index(t)
        old = tab[i].astype(np.float64)
        cos = float(unit(pa) @ unit(np.where(filt, old, 0.0)))
        mix = W_PA * unit(pa) + (1 - W_PA) * unit(np.where(filt, old, 0.0))
        changed[t] = dict(raw=mix, cos=cos, mins=mins, wts=wts,
                          selfs=selfs, metas=metas, ncov=len(rows),
                          old_norm=float(np.linalg.norm(old[filt])))
        print(f"  {t:9s} USE  n_cov5k={len(rows)} self={mins:+.2f} "
              f"w={np.round(wts / wts.sum(), 2)} cos(PA,old)={cos:+.3f}")

    print(f"\nchanged {len(changed)} rows, skipped {len(skipped)}")
    print("  skipped:", sorted(skipped))

    # 尺度: 组内 unit-mix 的自然范数中位对齐 raw17 锚
    n_raw = np.median([np.linalg.norm(tab[panel.index(t)][filt]) for t in raw_set])
    nats = np.array([np.linalg.norm(v["raw"][filt]) for v in changed.values()])
    c = n_raw / np.median(nats)
    print(f"raw17 anchor {n_raw:.2f}, mix natural median {np.median(nats):.3f} -> c {c:.3f}")

    for t, v in changed.items():
        i = panel.index(t)
        tab[i] = np.clip(c * v["raw"], -4, 4).astype(np.float32)

    keep = [i for i, t in enumerate(panel) if t not in changed]
    assert np.array_equal(tab[keep], v10["lfc"][keep]), "untouched rows changed!"
    print(f"v11 table {tab.shape}, max|.| {np.abs(tab).max():.2f}")
    newn = np.array([np.linalg.norm(tab[panel.index(t)][filt]) for t in changed])
    print(f"changed row filt-norm: median {np.median(newn):.2f}, "
          f"min {newn.min():.2f}, max {newn.max():.2f}")
    np.savez(OUT, names=np.array(panel), genes=genes, lfc=tab)
    print(f"saved {OUT}")

    json.dump({t: dict(cos=v["cos"], n_cov=int(v["ncov"]),
                       n_comp=int(len(v["metas"])),
                       mins=float(v["mins"]),
                       pids=[m["pid"] for m in v["metas"]],
                       tissues=[m["tissue"] for m in v["metas"]],
                       old_norm=v["old_norm"],
                       new_norm=float(np.linalg.norm(tab[panel.index(t)][filt])))
               for t, v in changed.items()},
              open("outputs/v11_pa_changes.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
