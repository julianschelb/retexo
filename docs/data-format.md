# Data Format

The predicted edit scripts are published as a dataset on the Hugging Face Hub:
[`julian-schelb/latin-classical-intertextuality-edit-scripts`](https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts).

It holds one record for each of the 1,490 references of the Loci Similes benchmark. The references are split into the
benchmark's five folds, `fold_0` to `fold_4`. The script of a reference was written by a model that was **not** trained on
that reference's fold, so every record is a true held-out prediction.

## A record

```json
{
  "id": "p0001",
  "benchmark_id": 1,
  "fold": 4,
  "reference_type": "cf.",
  "source": {"author": "verg", "work": "verg. aen.", "citation": "<verg. aen. 6.847.1>", "tokens": ["Excudent", "alii", "..."]},
  "reuse":  {"author": "hier", "work": "hier. epist.", "citation": "<hier. epist. 117.7.1.1>", "tokens": ["Legimus", "in", "..."]},
  "links": [
    {"reuse": 5, "source": 2, "label": "COPY", "relation": null, "confidence": 0.9994}
  ],
  "frame": [],
  "insertions": [0, 1, 2, 3, 4, 6],
  "deletions": [0, 1, 3, 5, 6]
}
```

| Field | Meaning |
|---|---|
| `id` | the pair's identifier |
| `benchmark_id` | the reference's id in the Loci Similes labels dataset |
| `fold` | the benchmark fold of the reusing passage |
| `reference_type` | `cit.` (citation) or `cf.` (allusion), as in the benchmark |
| `source`, `reuse` | the earlier and the later passage: author, work, citation, and the words |
| `links` | one entry per linked reuse word |
| `frame` | citing formulas such as *ut ait Maro*, as half-open ranges `[start, end)` of reuse words |
| `insertions` | reuse words that are neither linked nor part of a citing formula |
| `deletions` | source words that no link claims |

All indices are positions in the `tokens` of the passage they belong to, counted from 0.

### Links

| Field | Meaning |
|---|---|
| `reuse` | index of the reuse word |
| `source` | index of the source word it takes up |
| `label` | `COPY`, `INFLECT`, `SUBST`, `SPLIT` or `MERGE`, see [Labels](labels.md) |
| `relation` | for a `SUBST`, the relation between the two words when the model names one (`SYN`, `POS`, `NE-SUB`, ...), else `null` |
| `confidence` | the model's probability for this link |

Two details of the encoding:

- A `SPLIT` is two reuse words sharing one source word, so the source index appears in two links.
- A `MERGE` link points at the first of the two source words; the second counts as a deletion.

## Loading

```python
from datasets import load_dataset

scripts = load_dataset("julian-schelb/latin-classical-intertextuality-edit-scripts")
print(scripts)                       # fold_0 ... fold_4
pair = scripts["fold_4"][0]
```

Or with pandas, after downloading the Parquet files:

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/julian-schelb/latin-classical-intertextuality-edit-scripts/data/fold_4.parquet")
```

## Producing the same format

`retexo export` converts the raw `predictions.jsonl` that a run writes into this format, see the [CLI](cli.md). The
format is versioned in `retexo.export.SCHEMA_VERSION`.

## What these predictions are not

They are model output, not annotation. Most of the model's mistakes on inflections and substitutions are words it left
without a link, so a missing link is the first thing to suspect. Read the scripts as a first reading to correct, which is
how they are used in the paper, and not as ground truth.
