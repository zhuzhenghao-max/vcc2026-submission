"""Generate a submission h5ad from a per-target log2 fold-change override table.

This is the inference path used for the final 2026 Virtual Cell Challenge
entry. Every target in the submitted panel was covered by the override table,
so no trained model is needed. Each target response is read from the table,
mapped onto the context gene axis, and turned into counts by multinomial
sampling from that context's control profile.

Override table: an .npz file with

    names : (P,) target gene symbols
    genes : (G,) genes on the response axis
    lfc   : (P, G) log2 fold changes

Contexts: a CSV with header context,path. Each path is a control h5ad.
Targets: a CSV with a target_gene column.

Output: an h5ad with obs columns target_gene and context.
"""

import argparse
import os
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CLIP = 4.0


def load_override(path):
    z = np.load(path, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    genes = np.array([str(x) for x in z["genes"]])
    lfc = np.asarray(z["lfc"], dtype=np.float32)
    if lfc.shape != (len(names), len(genes)):
        raise SystemExit(
            f"{path}: lfc shape {lfc.shape} does not match "
            f"{len(names)} targets x {len(genes)} genes"
        )
    if not np.isfinite(lfc).all():
        raise SystemExit(f"{path}: override table contains NaN or inf")
    return names, genes, lfc


def gen_cells(ctrl_X, lfc, n_cells, umi_target, rng):
    """Sample counts from the control profile shifted by an lfc vector."""
    rates = np.clip(np.asarray(ctrl_X.mean(0)).ravel(), 0, None)
    rates = rates * np.exp2(np.clip(np.nan_to_num(lfc), -CLIP, CLIP))
    total = rates.sum()
    if total > 0:
        rates = rates / total
    else:
        rates = np.full(rates.shape, 1.0 / rates.size)
    X = rng.multinomial(umi_target, rates, size=n_cells).astype(np.float32)
    return sparse.csr_matrix(X)


def write_parts(contexts, targets, ov_names, ov_genes, ov_lfc,
                n_cells, seed, out):
    rng = np.random.default_rng(seed)
    part_dir = out + ".parts"
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    files = []

    ov_pos = {g: i for i, g in enumerate(ov_names)}
    missing = [g for g in targets if g not in ov_pos]
    if missing:
        raise SystemExit(
            f"{len(missing)} targets are absent from the override table: "
            f"{missing[:10]}"
        )

    for _, row in contexts.iterrows():
        ctx = str(row["context"])
        a = ad.read_h5ad(row["path"])
        if a.n_obs == 0:
            raise SystemExit(f"{row['path']}: no control cells")
        umi_target = int(np.median(np.asarray(a.X.sum(1)).ravel()))
        if umi_target <= 0:
            raise SystemExit(f"{row['path']}: median total counts is zero")
        pos = {g: i for i, g in enumerate(a.var_names.astype(str))}
        col = np.array([pos.get(g, -1) for g in ov_genes])
        ok = col >= 0
        n_mapped = int(ok.sum())

        for target in targets:
            lfc = np.zeros(a.n_vars, dtype=np.float32)
            take = ov_pos[target]
            lfc[col[ok]] = ov_lfc[take][ok]
            X = gen_cells(a.X, lfc, n_cells, umi_target, rng)
            obs = pd.DataFrame(
                {"target_gene": target, "context": ctx},
                index=[f"{ctx}_{target}_{k}" for k in range(n_cells)],
            )
            part = ad.AnnData(X=X, obs=obs)
            part.var_names = a.var_names
            p = os.path.join(part_dir, f"{ctx}_{target}.h5ad")
            part.write_h5ad(p)
            files.append(p)
        print(
            f"{ctx}: {len(targets)} targets x {n_cells} cells, "
            f"umi {umi_target}, override genes mapped {n_mapped}/{len(ov_genes)}"
        )

    if hasattr(ad, "experimental") and hasattr(ad.experimental, "concat_on_disk"):
        ad.experimental.concat_on_disk(files, out, axis=0, join="outer")
    else:
        merged = ad.concat([ad.read_h5ad(p) for p in files], join="outer")
        merged.write_h5ad(out)
    shutil.rmtree(part_dir, ignore_errors=True)
    print(f"wrote {out} from {len(files)} parts")


def main():
    p = argparse.ArgumentParser(
        description="Generate a VCC-format h5ad from an override table."
    )
    p.add_argument("--override", required=True,
                   help="npz with names, genes, lfc")
    p.add_argument("--contexts", required=True,
                   help="csv with header context,path (control h5ad per row)")
    p.add_argument("--targets", required=True,
                   help="csv with a target_gene column")
    p.add_argument("--n-cells", type=int, default=400,
                   help="cells per target per context (default: 400)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output h5ad")
    args = p.parse_args()

    ov_names, ov_genes, ov_lfc = load_override(args.override)
    contexts = pd.read_csv(args.contexts)
    targets = pd.read_csv(args.targets)["target_gene"].astype(str).tolist()
    print(f"override: {len(ov_names)} targets on {len(ov_genes)} genes")
    write_parts(contexts, targets, ov_names, ov_genes, ov_lfc,
                args.n_cells, args.seed, args.out)


if __name__ == "__main__":
    main()
