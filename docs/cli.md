# CLI Reference

```bash
retexo [--version] <command>
```

## `retexo export`

Converts the raw `predictions.jsonl` files of a run, one per test fold, into the [released format](data-format.md).

```bash
retexo export runs/ours_f0/predictions.jsonl runs/ours_f1/predictions.jsonl \
              runs/ours_f2/predictions.jsonl runs/ours_f3/predictions.jsonl \
              runs/ours_f4/predictions.jsonl --out export/ --format parquet
```

| Argument | Meaning |
|---|---|
| `predictions` | raw `predictions.jsonl` files, one per test fold |
| `--out` | output folder; writes `fold_<k>.jsonl` or `fold_<k>.parquet` |
| `--format` | `jsonl` (default) or `parquet` (needs `pyarrow`) |

The fold is read from the records, so the order of the files does not matter. A file that mixes folds is refused, and so
is a record that contradicts itself, for example a deletion that a link claims.

## `retexo-gui`

A Gradio demo: two Latin passages in, an edit script and an alignment out. It runs with the symbolic typer alone, or with
a trained typed-pointer checkpoint. It needs the `gui` extra.

```bash
pip install "retexo[gui] @ git+https://github.com/julianschelb/retexo"
retexo-gui                                  # the symbolic typer alone
retexo-gui --typed-pointer runs/ours_f4     # with a trained checkpoint
```

It listens on `127.0.0.1:7860` by default; `--host 0.0.0.0` exposes it to the network.

## Annotation app

A Gradio app in which an annotator corrects pre-annotated scripts word by word, with the model's proposal and its
confidence beside every cell. It reads the word-level annotation from `$RETEXO_HOME/data/gold_full`, which is not part of
this repository, and writes the corrections per annotator.

```bash
python -m retexo_gui.annotate --annotator NAME
```
