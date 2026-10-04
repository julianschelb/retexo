# Retexo

**Word-level explanations of text reuse in Latin literature.**

A later author who quotes or alludes to an earlier one has done something to the words: kept them, changed their
inflection, replaced them. Retexo writes that down. For a pair of passages it first aligns every word of the reusing
passage to the source word it takes up, or to none, and then names the operation behind each link. The result is an
*edit script*, a set of labeled links between two passages that a philologist can read and contest.

Retexo learns these scripts from very little annotation. Synthetic pairs, built by applying known operations to Latin
text, teach a pointer network the task before a single pair is annotated, and active learning chooses the few pairs an
annotator corrects.

## What is here

- **[Predicted edit scripts](data-format.md)** for all 1,490 references of the
  [Loci Similes](https://huggingface.co/collections/julian-schelb/datasets-for-latin-intertextuality-search) benchmark,
  published on the Hugging Face Hub.
- **[A review app](https://julianschelb.github.io/retexo/demo/)** to browse the script of every reference, drawn as arrows between the two passages.
- **The `retexo` package**: the model, the synthetic-pair generator, the active-learning loop, the baselines it is
  compared with, and the scorer.

## Quick start

```bash
pip install "retexo @ git+https://github.com/julianschelb/retexo"
```

Read the predictions, with the Hugging Face `datasets` library:

```python
from datasets import load_dataset

scripts = load_dataset("julian-schelb/latin-classical-intertextuality-edit-scripts", split="fold_4")
pair = scripts[0]

for link in pair["links"]:
    print(pair["reuse"]["tokens"][link["reuse"]], "<-", pair["source"]["tokens"][link["source"]], link["label"])
```

## Documentation

- [Getting Started](getting-started.md): installation and where retexo looks for data
- [Data Format](data-format.md): the released predictions, field by field
- [Labels](labels.md): the operations and what they mean
- [CLI Reference](cli.md): `retexo export` and the annotation app
- [Review App](web-demo.md): the web app and how it is built
- [API Reference](api/index.md)
- [Development](development.md)

## Authors

- **Julian Schelb**, University of Konstanz
- **Michael Wittweiler**, University of Zurich
- **Marie Revellio**, University of Konstanz

## Citation

The paper is in preparation, and its reference will be added here. Retexo builds on the
[Loci Similes](https://arxiv.org/abs/2601.07533) benchmark and its package,
[locisimiles](https://julianschelb.github.io/locisimiles/).

## License

MIT
