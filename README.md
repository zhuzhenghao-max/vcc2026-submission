# Virtual Cell Challenge 2026: cross-line perturbation response prediction

Solo entry to the 2026 Virtual Cell Challenge. The task was to predict
single-cell transcriptome responses to CRISPR perturbations in unseen cell
contexts. The submission placed **392nd overall on the final hidden-test
leaderboard** with an overall score of **0.106045**.

The method is deliberately simple. Rather than training a generative model,
it assembles each predicted response from perturbation measurements that
already exist in public cell lines, mixes them at the level of unit-normalized
response vectors, and converts the result into counts from the target
context's control profile. On this benchmark that retrieval-style approach
outperformed the tested GEARS and scGPT configurations.

## Results

Final submission (`lfcmap-c10-ovlfinal-pa`, entry `Z7LscjnKvbuKBh4gLWdI`,
submitted 2026-10-31, contexts D/E/F):

| Metric | Value |
|---|---|
| Rank (final leaderboard) | 392 |
| Overall | 0.106045 |
| Perturbation discrimination score (PDS) | 0.553948 |
| Expression MSE component | 0 |
| DE log-FC NMAE | 0.054324 |
| DE direction fidelity | -0.030712 |
| DE direction reach | 0.051782 |
| DE significance Jaccard | 0.006928 |

Best validation-phase entry (entry `S1N4fs8QFBKNuqXIj3xm`, contexts A/B/C):
overall **0.112010**, PDS **0.582614**.

The accompanying benchmark paper studies why this kind of mixing works and
when it stops working:

- Repository: <https://github.com/zhuzhenghao-max/mixing-vs-selection>
- Archive: <https://doi.org/10.5281/zenodo.24223291>
- Title: *Separating mixing from selection: a leave-one-line-out benchmark
  and an audit protocol for cross-cell-line perturbation evaluation*

## Method

For each target gene, the response is assembled by a priority cascade.

1. **Measured H1 responses.** Targets measured in the released 2025 Virtual
   Cell Challenge H1 training/validation splits use their measured response:
   mean-CPM log2 ratio against non-targeting controls, genes below 1 CPM set
   to zero, values clipped to +/-4. Seventeen panel targets entered this way.
2. **Cross-line measured mixes.** Remaining targets use Replogle genome-wide
   Perturb-seq (K562) and X-Atlas/Orion (HCT116 and HEK293T). Responses are
   unit-normalized and mixed as 0.4 x Orion + 0.6 x GWPS. Groups with only
   one source use that source. Each group is rescaled so its median response
   norm matches the measured H1 rows.
3. **Embedding fallback.** Targets without the above use an embedding-based
   response donor, rescaled the same way. Remaining targets receive a zero
   row.
4. **PerturbAtlas gate (final submission).** For 19 non-raw targets,
   PerturbAtlas knockdown/knockout differential-expression tables were
   filtered to comparisons with at least 5,000 cells. A target was admitted
   only if at least one comparison showed a self-knockdown log2FC at or below
   -1. Admitted rows were averaged as unit vectors weighted by
   max(0, -self log2FC) and blended as 0.4 x PerturbAtlas + 0.6 x the
   existing row. Measured H1 rows, embedding rows, and zero rows were left
   untouched.

The resulting per-target log2 fold-change table is mapped onto each target
context's gene axis. For every target and context, gene rates are the
control mean expression multiplied by 2^lfc; rates are normalized and 400
cells are sampled from a multinomial at the context's median UMI count.

The final table covered all 300 panel targets, so no target was left for the
ridge model to predict. The submitted run used
`linear_baseline.py predict --lfc-override`; the ridge weights were loaded
but had no effect on the output. `src/predict_override.py` is a trimmed
equivalent with that unused path removed. `src/linear_baseline.py` is
included for reference.

## Repository layout

```
src/predict_override.py   override-table to submission h5ad (final path)
src/linear_baseline.py    development predictor (ridge + override option)
src/prior_features.py     prior features imported by linear_baseline.py
build/build_ov10.py       builds the v10 override table (reference)
build/build_ov_final.py   builds the final-panel table from v10
build/build_ov11.py       adds gated PerturbAtlas rows (final table)
build/README.md           build-chain inputs and ordering
configs/                  example context CSV
data/README.md            required public data and expected layout
```

## Running inference

Install the dependencies:

```
pip install -r requirements.txt
```

Prepare a context CSV and a target CSV:

```
# configs/contexts.csv
context,path
D,/path/to/context_D.h5ad
E,/path/to/context_E.h5ad
F,/path/to/context_F.h5ad

# targets.csv
target_gene
GENE1
GENE2
...
```

Generate a prediction:

```
python src/predict_override.py \
  --override /path/to/lfc_override_2026_v11.npz \
  --contexts configs/contexts.csv \
  --targets /path/to/targets.csv \
  --n-cells 400 \
  --out predictions.h5ad
```

The output has `obs` columns `target_gene` and `context`, and one row per
cell. Package and submit with the challenge CLI (0.3.0 was used):

```
vcc prep predictions.h5ad -g gene_names_headerless.csv --perts pert_counts.csv \
  -o predictions.vcc
vcc submit predictions.vcc -m "model name" -d "description"
```

The challenge is closed, so new submissions are no longer scored.

## Rebuilding the override table

`build/build_ov_final.py` reads a v10 override table and the public H1,
Replogle, X-Atlas/Orion, and embedding assets. `build/build_ov11.py` then
adds the PerturbAtlas rows. The expected inputs and outputs are listed in
`build/README.md`.

The upstream tables (v1 through v10) are not included here. They were built
by the earlier scripts in the same project and depend on derived caches such
as the 18,080-gene axis cache and the H1 raw-response table. The companion
benchmark repository contains much of that development code. Neither this
repository nor the archive redistributes the submitted prediction file, the
derived override tables, or the challenge reference contexts.

## Data

Only public data were used:

| Data | Role |
|---|---|
| 2025 Virtual Cell Challenge H1 training and validation splits | Measured H1 rows |
| Replogle et al. (2022), genome-wide Perturb-seq (K562) | Cross-line mix |
| X-Atlas/Orion (HCT116, HEK293T) | Cross-line mix |
| PerturbAtlas (<https://perturbatlas.kratoss.site>) | Gated final rows |
| HGNC complete set (<https://www.genenames.org/download/archive/>) | Ensembl-to-symbol mapping |

No challenge test data or test labels were used. The H1 responses come from
the released training split. The final contexts were scored externally by
the organizers.

## Notes and limitations

- This is a retrieval-heavy baseline. Its performance depends on the
  availability of a measured response for the same target in another cell
  line; it is not a solution for targets with no measured response anywhere.
- The PerturbAtlas gate was chosen after observing H1 validation gains. It
  is a competition choice, not an independently validated rule.
- The final leaderboard score is lower than the best validation score
  (0.106045 versus 0.112010). The two partitions are not directly comparable;
  the hidden D/E/F contexts were scored only once, at the end.
- The exact submission cannot be regenerated from this repository alone
  because the final override table and the challenge reference contexts are
  not redistributed.

## AI assistance

Claude Code (Anthropic) was used as a coding and analysis assistant for
implementation, data processing, debugging, and figure and manuscript
editing. The author set the research direction and method choices, directed
the experiments, and verified every submitted result.

## License

Code is released under the MIT license. The method description and results
may be cited with attribution.
