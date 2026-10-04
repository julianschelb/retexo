# Retexo

[![CI](https://github.com/julianschelb/retexo/actions/workflows/ci.yml/badge.svg)](https://github.com/julianschelb/retexo/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-julianschelb.github.io%2Fretexo-blue)](https://julianschelb.github.io/retexo/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-edit--scripts-yellow)](https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts)
[![PyPI](https://img.shields.io/pypi/v/retexo)](https://pypi.org/project/retexo/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/julianschelb/retexo/blob/main/LICENSE)

**Word-level explanations of text reuse in Latin literature.**

When a later author quotes or alludes to an earlier one, the interesting question is *how* the source was reworked.
Retexo answers it word by word: it aligns every word of the reusing passage to the source word it takes up, or to none,
and names the operation behind each link, such as a copy, a change of inflection, or a substitution. The result is an
*edit script*: a set of labeled links between two passages that a philologist can read and contest.

Retexo learns these scripts from very little annotation. Synthetic pairs, built by applying known operations to Latin
text, teach a pointer network the task before a single pair is annotated, and active learning chooses the few pairs an
annotator corrects.

- **Predicted edit scripts** for all 1,490 references of the [Loci Similes](https://arxiv.org/abs/2601.07533) benchmark:
  [Hugging Face dataset](https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts)
- **Review app** to browse the script of every reference, drawn as arrows between the passages:
  [julianschelb.github.io/retexo/demo](https://julianschelb.github.io/retexo/demo/)
- **Documentation:** [julianschelb.github.io/retexo](https://julianschelb.github.io/retexo/)

## Quick start

```bash
pip install retexo
```

Read the predictions, with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

scripts = load_dataset("julian-schelb/latin-classical-intertextuality-edit-scripts", split="fold_4")
pair = scripts[0]

for link in pair["links"]:
    print(pair["reuse"]["tokens"][link["reuse"]], "<-", pair["source"]["tokens"][link["source"]], link["label"])
```

Convert the raw predictions of a run into the same format:

```bash
retexo export runs/f0/predictions.jsonl runs/f1/predictions.jsonl runs/f2/predictions.jsonl \
              runs/f3/predictions.jsonl runs/f4/predictions.jsonl --out export/ --format parquet
```

## Labels

| Label | Meaning |
|---|---|
| `COPY` | the same form, spelling variants folded |
| `INFLECT` | the same lemma in another form |
| `SUBST` | another lemma in the same slot, optionally with a named relation such as a synonym |
| `SPLIT`, `MERGE` | two reuse words for one source word, or one for two |
| `INS`, `FRAME`, `DEL` | an unlinked reuse word, a word of a citing formula, an unclaimed source word |

See the [documentation](https://julianschelb.github.io/retexo/labels/) for the conventions.

## Repository layout

```
src/retexo/         the package: model, generator, active learning, baselines, scorer
src/retexo_gui/     Gradio demo and annotation app (gui extra)
tests/              pytest suite
examples/           notebooks on the operations, the generator, the passage class and the decoder
docs/               documentation (MkDocs)
webapp/             the review app (React, Vite), published under /demo/
```

The experiments behind the paper live in a separate repository that installs retexo as a dependency.

## Development

```bash
pip install -e ".[dev,lexical]"
pytest
mkdocs serve
```

See [docs/development.md](https://julianschelb.github.io/retexo/development/).

## Authors

Julian Schelb (University of Konstanz), Michael Wittweiler (University of Zurich), Marie Revellio (University of Konstanz).

## License

MIT
