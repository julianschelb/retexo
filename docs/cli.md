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

The annotation and review app, which needs the `gui` extra:

```bash
pip install "retexo[gui] @ git+https://github.com/julianschelb/retexo"
retexo-gui
```
