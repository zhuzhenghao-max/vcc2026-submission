"""先验基因集特征: TF-靶基因/通路成员注释在对照上的共表达与表达聚合

Enrichr geneSetLibrary 文本格式: 集名<TAB>空<TAB>基因<TAB>基因...
特征维度: TF 靶基因聚合 42 维 + Hallmark/KEGG 成员指示
"""

import re

import numpy as np

AGG_DIM = 42  # 集合大小 + log1p 平均表达 + 40 维 PC 载荷均值


def parse_lib(path):
    sets = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        name = parts[0]
        genes = [g for g in parts[2:] if g]
        if re.search(r"\b(mouse|rat|yeast|worm|fly|zebrafish)\b", name):
            continue
        if genes:
            sets[name] = set(genes)
    return sets


def prior_agg(gene_sets, genes, mean_expr, pc):
    """对每个基因: 它作为 TF 的靶基因集合聚合 (非 TF 记 0); 或所在集的成员指示"""
    pos = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    out = np.zeros((n, AGG_DIM))
    for tf, targets in gene_sets.items():
        if tf not in pos:
            continue
        idx = [pos[g] for g in targets if g in pos]
        if not idx:
            continue
        i = np.array(idx)
        out[pos[tf]] = np.hstack([len(i), np.log1p(mean_expr[i].mean()), pc[i].mean(0)])
    return out


def set_indicators(gene_sets, genes, max_sets=None):
    """每基因的集合成员指示, 列 = 基因集 (按名排序, 可截断)"""
    names = sorted(gene_sets)
    if max_sets:
        names = names[:max_sets]
    pos = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    out = np.zeros((n, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        for g in gene_sets[name]:
            if g in pos:
                out[pos[g], j] = 1.0
    return out


def build_priors(ctrl_lognorm, pc, genes, tf_paths, path_paths):
    """TF 靶基因聚合 (42 维) + 通路成员指示, 基因轴与 genes 对齐"""
    X = ctrl_lognorm.X
    mean = np.asarray(X.mean(0)).ravel()
    tf_union = {}
    for path in tf_paths:
        for tf, targets in parse_lib(path).items():
            tf_union.setdefault(tf, set()).update(targets)
    F_tf = prior_agg(tf_union, genes, mean, pc)
    sets = {}
    for path in path_paths:
        sets.update(parse_lib(path))
    F_ind = set_indicators(sets, genes)
    return np.hstack([F_tf, F_ind])
