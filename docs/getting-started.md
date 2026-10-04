# Getting Started

## Installation

Retexo needs Python 3.10 or later.

```bash
pip install "retexo @ git+https://github.com/julianschelb/retexo"
```

Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `lexical` | CLTK, spaCy, Stanza, NLTK, gensim | the symbolic typer and the lexical evidence of the model |
| `llm` | Anthropic client, PEFT | the language-model baselines |
| `gui` | Gradio | the Gradio demo and the annotation app |
| `plots` | Matplotlib | learning-curve plots |

```bash
pip install "retexo[lexical] @ git+https://github.com/julianschelb/retexo"
```

For development, see [Development](development.md).

## Where retexo looks for data

Everything that is not code, the benchmark and annotation files and the resource caches, lives under one directory. It is the working directory unless `RETEXO_HOME` points elsewhere:

```
$RETEXO_HOME/
  data/processed/        the Loci Similes benchmark
  data/gold_full/        word-level annotation (not part of this repository)
  resources_cache/       cached Latin WordNet responses
```

## Reading the predictions

You do not need the code to read the released predictions; see [Data Format](data-format.md). The code is for
producing predictions and retraining the model.

## First check

```bash
retexo --version
retexo export --help
```
