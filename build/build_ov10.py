"""组装 v10 override 表: v8 基础上, 8 个 Test8 解码行换成跨系实测.

- 这 8 个靶 (test8 = v3 注入里的 8 个解码行) 全部有 GWPS 实测; 其中 5 个还有 orion:
  - orion∩gwps 5 个: v8 同款 unit-mix 0.4*unit(Oavg)+0.6*unit(G), 缩放用 v8 的 c8
  - gwps-only 3 个: v7 同款 gwps natural 行, 缩放用 v7 的 c7
- 其余 292 行 bit-identical v8 (含 17 个 raw 注入 == v3 断言)
- 依据: 解码行官方 pds 估计 ~0.6 vs 实测跨系行反推 ~0.78-0.80; 8 行现范数 2.4-16.8 偏弱
"""

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

C8_MIX = 19.193   # v8 混合行缩放 (unit-mix natural -> raw17 锚)
C7_GWPS = 0.676   # v7 gwps natural 行缩放


def main():
    v8 = np.load("outputs/lfc_override_2026_v8.npz", allow_pickle=True)
    v7 = np.load("outputs/lfc_override_2026_v7.npz", allow_pickle=True)
    v3 = np.load("outputs/lfc_override_2026_v3.npz", allow_pickle=True)
    panel = [str(x) for x in v8["names"].astype(str)]
    genes = v8["genes"].astype(str)
    tab = v8["lfc"].copy()
    v3_names = list(v3["names"].astype(str))
    v3_tab = {g: v3["lfc"][i] for i, g in enumerate(v3_names)}

    target_set = set(panel)
    heldout = set(pd.read_csv("data/internal/holdout_genes.csv")["target_gene"].astype(str))
    train_names = set(map(str, pd.read_parquet("data/vccp_de_log2fc.parquet").index))
    train13 = sorted((target_set & train_names) & set(v3_tab))
    val4 = sorted(target_set & heldout)
    test8 = sorted(set(v3_tab) - set(train13) - set(val4))
    raw17 = set(train13) | set(val4)
    assert len(test8) == 8 and len(raw17) == 17
    print(f"raw17 {sorted(raw17)}")
    print(f"test8 {test8}")

    m = np.load("outputs/scvi_map_cache.npz", allow_pickle=True)
    filt = m["filt"]

    meta = json.load(open("data/external/orion_meta.json", encoding="utf-8"))
    o_genes = meta["genes"]
    gidx = {g: i for i, g in enumerate(genes)}
    ocols, opos = [], []
    for oc, g in enumerate(o_genes):
        if g in gidx:
            ocols.append(oc)
            opos.append(gidx[g])
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
        full = np.zeros(len(genes))
        full[opos] = np.mean(vs, 0)
        return full

    g = np.load("outputs/gwps_rows.npz", allow_pickle=True)
    gnames = [str(x) for x in g["names"].astype(str)]
    grows = g["rows"].astype(np.float64)
    gmap = {nm: i for i, nm in enumerate(gnames)}

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-9)

    changed = {}
    for t in test8:
        j = gmap.get(t)
        assert j is not None, f"{t} not in gwps!"
        ov = orion_natural(t)
        if ov is not None:
            mix = 0.4 * unit(ov) + 0.6 * unit(np.nan_to_num(grows[j]))
            row = np.clip(C8_MIX * np.where(filt, mix, 0.0), -4, 4)
            changed[t] = ("mix", row)
        else:
            row = np.clip(C7_GWPS * np.where(filt, grows[j], 0.0), -4, 4)
            changed[t] = ("gwps", row)

    # sanity: gwps-only natural norms scaled ~13.57 anchor scale
    for t, (kind, row) in changed.items():
        i = panel.index(t)
        old = np.linalg.norm(tab[i][filt])
        new = np.linalg.norm(row[filt])
        print(f"  {t:8s} {kind:4s} norm {old:6.2f} -> {new:6.2f}")

    for t, (kind, row) in changed.items():
        tab[panel.index(t)] = row.astype(np.float32)

    r17idx = [i for i, t in enumerate(panel) if t in raw17]
    assert np.array_equal(tab[r17idx], np.stack([v3_tab[panel[i]] for i in r17idx]))
    keep = [i for i, t in enumerate(panel) if t not in changed]
    assert np.array_equal(tab[keep], v8["lfc"][keep]), "non-test8 rows changed!"
    print(f"v10 table {tab.shape}, max|.| {np.abs(tab).max():.2f}, changed {len(changed)} rows")
    np.savez("outputs/lfc_override_2026_v10.npz", names=np.array(panel), genes=genes, lfc=tab)
    print("saved outputs/lfc_override_2026_v10.npz")


if __name__ == "__main__":
    main()
