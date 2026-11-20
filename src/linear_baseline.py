"""线性零样本基线: 基因共表达特征 -> 扰动 lfc 向量的多变量岭回归

训练: F(基因特征) 来自训练细胞系的对照, D(lfc) 来自其扰动
零样本: F 从目标上下文对照重新计算, D = F @ W + b, 按 2^d 倍率多项式重采样细胞
输出对齐上传契约: 稀疏计数, 每扰动固定细胞数, 逐扰动写盘后 concat_on_disk 合并
"""

import argparse
import os
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from prior_features import build_priors

CONTROL = "non-targeting"
N_FEAT_PC = 40
CLIP = 4.0
EPS = 1e-9


def lognorm_cpm(a):
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e6)
    sc.pp.log1p(b)
    return b


def gene_features(ctrl_lognorm, n_pcs=N_FEAT_PC):
    """每基因特征: 共表达 PCA 载荷 + 表达统计"""
    sc.pp.pca(ctrl_lognorm, n_comps=n_pcs, svd_solver="arpack")
    pc = np.asarray(ctrl_lognorm.varm["PCs"])
    X = ctrl_lognorm.X
    X2 = X.power(2) if hasattr(X, "power") else np.square(X)
    mean = np.asarray(X.mean(0)).ravel()
    var = np.asarray(X2.mean(0)).ravel() - mean**2
    det = np.asarray((X > 0).mean(0)).ravel()
    stats = np.stack([np.log1p(mean), np.log1p(np.clip(var, 0, None)), det], 1)
    return np.hstack([pc, stats])


def mean_cpm(a):
    """每细胞归一到 1e6 后的均值, 与 cell-eval2 的 lfc 定义一致"""
    X = a.X
    tot = np.asarray(X.sum(1)).ravel()
    tot = np.where(tot > 0, tot, 1.0)
    Xn = X.multiply(1e6 / tot[:, None]) if sparse.issparse(X) else X * (1e6 / tot)[:, None]
    return np.asarray(Xn.mean(0)).ravel()


def responses(pert, ctrl, pert_col, max_cells=None, rng=None):
    base = mean_cpm(ctrl)
    out = {}
    labels = pert.obs[pert_col].astype(str).to_numpy()
    for p in sorted(set(labels) - {CONTROL}):
        idx = np.where(labels == p)[0]
        if max_cells and len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        # backed 切片转内存, 每扰动几十 MB, 全量矩阵不会进内存
        m = mean_cpm(pert[idx].to_memory())
        out[p] = np.log2((m + EPS) / (base + EPS))
    return out


def fit_ridge(F, D, lam):
    """w = (F'F + lam I)^-1 F' D, 截距不惩罚"""
    n = F.shape[0]
    F1 = np.hstack([F, np.ones((n, 1))])
    A = F1.T @ F1 + lam * np.eye(F1.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, F1.T @ D)
    return W[:-1], W[-1]


def gen_cells(ctrl, lfc, n_cells, umi_target, rng):
    rates = np.clip(np.asarray(ctrl.X.mean(0)).ravel(), 0, None)
    rates = rates * np.exp2(np.clip(np.nan_to_num(lfc), -CLIP, CLIP))
    s = rates.sum()
    rates = rates / s if s > 0 else np.full(rates.shape, 1 / rates.size)
    X = rng.multinomial(umi_target, rates, size=n_cells).astype(np.float32)
    return sparse.csr_matrix(X)


def fit(args):
    rng = np.random.default_rng(args.seed)
    ctrl_full = ad.read_h5ad(args.ctrl, backed="r")
    if args.max_cells_ctrl and ctrl_full.n_obs > args.max_cells_ctrl:
        idx = rng.choice(ctrl_full.n_obs, args.max_cells_ctrl, replace=False)
        ctrl = ctrl_full[idx].to_memory()
    else:
        ctrl = ctrl_full.to_memory()
    ctrl_ln = lognorm_cpm(ctrl)
    F = gene_features(ctrl_ln, args.n_pcs)
    genes = ctrl.var_names.to_numpy().astype(str)
    if args.prior_tf or args.prior_path:
        pc = np.asarray(ctrl_ln.varm["PCs"])
        F = np.hstack([F, build_priors(ctrl_ln, pc, genes, args.prior_tf, args.prior_path)])
    pos = {g: i for i, g in enumerate(genes)}

    # 全量 fit 时扰动侧矩阵几十 GB, backed 逐扰动切片, 只留当前扰动在内存
    D_map = {}
    for path in args.perturbed.split(","):
        pert = ad.read_h5ad(path, backed="r")
        if not np.array_equal(pert.var_names.to_numpy().astype(str), genes):
            raise SystemExit(f"{path}: var 顺序与对照不一致, 不能直接拼 lfc")
        m = responses(pert, ctrl, args.pert_col, args.max_cells_per_pert, rng)
        dup = set(D_map) & set(m)
        if dup:
            raise SystemExit(f"duplicate perts across files: {sorted(dup)}")
        D_map.update(m)
        print(f"{path}: {len(m)} perts")

    h5ad_names = set(D_map)
    if args.lfc_table:
        # 外部 lfc 表 (deseq2 等), 行=扰动 列=基因, 直接当 D 行, 后面同样置零裁剪
        t = pd.read_parquet(args.lfc_table)
        if [str(c) for c in t.columns] != genes.tolist():
            raise SystemExit(f"{args.lfc_table}: 列与对照基因顺序不一致")
        dup = set(D_map) & set(t.index)
        if dup:
            raise SystemExit(f"lfc-table 与 h5ad 扰动重叠: {sorted(dup)}")
        if not np.isfinite(t.to_numpy()).all():
            raise SystemExit(f"{args.lfc_table}: 含 NaN/inf")
        for p in t.index:
            D_map[p] = t.loc[p].to_numpy().astype(np.float64)
        print(f"{args.lfc_table}: {t.shape[0]} perts")
    train_genes = sorted(D_map)
    missing = [g for g in train_genes if g not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} train targets not in panel: {missing[:10]}")

    D = np.stack([D_map[g] for g in train_genes])
    # 目标基因自己那一行是敲低本身, 六个指标都排除它, 置 0 免得污染回归
    D[np.arange(len(train_genes)), [pos[g] for g in train_genes]] = 0.0
    D = np.clip(D, -CLIP, CLIP)
    if args.standardize:
        fmean = F.mean(0)
        fstd = F.std(0)
        fstd = np.where(fstd > 0, fstd, 1.0)
        Fs = (F - fmean) / fstd
    else:
        fmean, fstd, Fs = np.zeros(0), np.zeros(0), F
    F_tr = np.stack([Fs[pos[g]] for g in train_genes])

    os.makedirs(args.outdir, exist_ok=True)
    W, b = fit_ridge(F_tr, D, args.lam)
    # 过滤基因 (对照均值>5 CPM) 上的典型范数, 预测时按该子集归一化
    base = mean_cpm(ctrl)
    filt = base > 5
    h5_idx = [i for i, g in enumerate(train_genes) if g in h5ad_names]
    lfc_norm = float(np.median([np.linalg.norm(D[i][filt]) for i in h5_idx])) if h5_idx else 0.0
    np.savez(os.path.join(args.outdir, "model.npz"), W=W, b=b, genes=genes,
             n_pcs=args.n_pcs,
             prior_tf=";".join(args.prior_tf),
             prior_path=";".join(args.prior_path),
             fmean=fmean, fstd=fstd, lfc_norm=lfc_norm)
    print(f"fit on {len(train_genes)} perts, W {W.shape}, lambda {args.lam}, lfc_norm {lfc_norm:.2f}")


def predict(args):
    m = np.load(os.path.join(args.model, "model.npz"))
    W, b = m["W"], m["b"]
    fit_genes = m["genes"].astype(str)
    n_pcs = int(m["n_pcs"]) if "n_pcs" in m else N_FEAT_PC
    prior_tf = [x for x in str(m["prior_tf"]).split(";") if x] if "prior_tf" in m else []
    prior_path = [x for x in str(m["prior_path"]).split(";") if x] if "prior_path" in m else []

    ctx = pd.read_csv(args.contexts)
    targets = pd.read_csv(args.targets)["target_gene"].tolist()
    rng = np.random.default_rng(args.seed)
    norm_tab = {}
    fallback_norm = 0.0
    if args.norm_table:
        t = np.load(args.norm_table)
        norm_tab = {g: float(v) for g, v in zip(t["genes"].astype(str), t["norms"])}
        fallback_norm = float(np.median(t["norms"]))
    blend_vec = None
    if args.lfc_blend:
        bl = np.load(args.lfc_blend)
        blg = bl["genes"].astype(str)
        fit_g = m["genes"].astype(str)
        blpos = {g: i for i, g in enumerate(blg)}
        blend_vec = np.zeros(len(fit_g), dtype=np.float64)
        for i, g in enumerate(fit_g):
            if g in blpos:
                blend_vec[i] = float(bl["lfc"][blpos[g]])
        print(f"blend loaded: {len(blg)} genes, alpha={args.alpha}")
    ov_tab, ov_lfc = None, None
    if args.lfc_override:
        ov = np.load(args.lfc_override)
        ov_tab = {g: i for i, g in enumerate(ov["names"].astype(str))}
        ov_lfc = ov["lfc"]
        print(f"override loaded: {len(ov_tab)} perts")

    part_dir = args.out + ".parts"
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    files = []
    for _, row in ctx.iterrows():
        a = ad.read_h5ad(row["path"])
        ctrl_ln = lognorm_cpm(a)
        F = gene_features(ctrl_ln, n_pcs)
        genes = a.var_names.to_numpy().astype(str)
        if prior_tf or prior_path:
            pc = np.asarray(ctrl_ln.varm["PCs"])
            F = np.hstack([F, build_priors(ctrl_ln, pc, genes, prior_tf, prior_path)])
        if "fmean" in m and len(m["fmean"]) > 0:
            F = (F - m["fmean"]) / m["fstd"]
        pos = {g: i for i, g in enumerate(genes)}
        missing = [g for g in targets if g not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} targets not in context {row['context']}: {missing[:10]}")
        umi_target = int(np.median(np.asarray(a.X.sum(1)).ravel()))
        base = mean_cpm(a)
        filt = base > 5
        # 效应向量的输出基因轴 = 训练面板, 目标上下文缺的基因 lfc 记 0
        keep = np.array([pos.get(g, -1) for g in fit_genes])
        kept = keep[keep >= 0]
        fit_ok = np.where(keep >= 0)[0]
        for g in targets:
            lfc = np.zeros(len(genes))
            if ov_tab is not None and g in ov_tab:
                lfc[kept] = ov_lfc[ov_tab[g]][fit_ok]
            else:
                lfc_fit = F[pos[g]] @ W + b
                if blend_vec is not None:
                    lfc_fit = args.alpha * lfc_fit + (1.0 - args.alpha) * blend_vec
                lfc[kept] = lfc_fit[fit_ok]
                if norm_tab:
                    n = np.linalg.norm(lfc[filt])
                    tgt = norm_tab.get(g, fallback_norm)
                    if n > 1e-9:
                        lfc = lfc * min(args.lfc_scale * tgt / n, 20.0)
                elif "lfc_norm" in m:
                    n = np.linalg.norm(lfc[filt])
                    if n > 1e-9:
                        lfc = lfc * min(args.lfc_scale * float(m["lfc_norm"]) / n, 20.0)
                else:
                    lfc = args.lfc_scale * lfc
            X = gen_cells(a, lfc, args.n_cells, umi_target, rng)
            obs = pd.DataFrame({"target_gene": g, "context": row["context"]},
                               index=[f"{row['context']}_{g}_{k}" for k in range(args.n_cells)])
            part = ad.AnnData(X=X, obs=obs)
            part.var_names = genes
            p = os.path.join(part_dir, f"{row['context']}_{g}.h5ad")
            part.write_h5ad(p)
            files.append(p)
        print(f"{row['context']}: {len(targets)} perts x {args.n_cells} cells, umi {umi_target}")

    ad.experimental.concat_on_disk(files, args.out, axis=0, join="outer")
    shutil.rmtree(part_dir)
    print(f"wrote {args.out} from {len(files)} parts")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--ctrl", required=True, help="训练细胞系对照 h5ad")
    f.add_argument("--perturbed", required=True,
                   help="训练扰动 h5ad, 逗号分隔可传多个, 逐文件切片不整载")
    f.add_argument("--pert-col", default="target_gene")
    f.add_argument("--lam", type=float, default=1.0)
    f.add_argument("--max-cells-per-pert", type=int, default=0,
                   help="每扰动随机取这么多细胞估均值, 0 表示不限制")
    f.add_argument("--max-cells-ctrl", type=int, default=0,
                   help="对照随机取这么多细胞算特征, 0 表示不限制")
    f.add_argument("--lfc-table", default="",
                   help="外部 lfc parquet (行=扰动 列=基因), 与 h5ad 来源合并训练")
    f.add_argument("--n-pcs", type=int, default=N_FEAT_PC)
    f.add_argument("--standardize", action="store_true",
                   help="特征列标准化到单位方差再回归")
    f.add_argument("--prior-tf", nargs="*", default=[],
                   help="TF-靶基因库 (Enrichr 文本格式)")
    f.add_argument("--prior-path", nargs="*", default=[],
                   help="通路库, 成员指示特征")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--outdir", required=True)
    f.set_defaults(fn=fit)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--contexts", required=True, help="csv: context,path 每行一个目标上下文对照")
    pr.add_argument("--targets", required=True, help="目标基因 csv (target_gene 列)")
    pr.add_argument("--n-cells", type=int, default=400)
    pr.add_argument("--lfc-scale", type=float, default=1.0,
                    help="预测 lfc 的整体缩放, 校准时用")
    pr.add_argument("--norm-table", default="",
                    help="npz: genes,norms 每基因 lfc 范数表 (per-gene 幅度校准)")
    pr.add_argument("--lfc-blend", default="",
                    help="npz: genes,lfc 混合底向量 (如 H1 平均响应); lfc = alpha*fit + (1-alpha)*blend")
    pr.add_argument("--alpha", type=float, default=1.0,
                    help="fit 分量权重; alpha=0 = 纯混合底向量 (均值响应仿制)")
    pr.add_argument("--lfc-override", default="",
                    help="npz: names,genes,lfc 面值覆盖指定扰动的 lfc (跳过缩放)")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", required=True)
    pr.set_defaults(fn=predict)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
