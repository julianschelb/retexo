# API Reference

The released predictions are read with the Hugging Face `datasets` library or plain JSON, see
[Data Format](../data-format.md). The Python API covers producing them.

| Module | Content |
|---|---|
| [`retexo.export`](export.md) | raw predictions to the released format |
| [`retexo.paths`](paths.md) | where retexo looks for data, caches and runs |

The rest of the package is organised by what a piece does:

| Package | Content |
|---|---|
| `retexo.core` | edit operations, the builder that makes a variant from a source, and the scribe that replays and verifies a script |
| `retexo.datasets` | the synthetic-pair generator and the records it writes |
| `retexo.aligners` | scoring and decoding of source-word choices |
| `retexo.baselines` | the systems compared in the paper, the decoder, and the scorer |
| `retexo.unified` | the stretch-and-link pointer |
| `retexo.edit_typing` | the symbolic typer that names a link from the two words |
| `retexo.resources` | Latin morphology, WordNet, named entities and word vectors |
| `retexo.pretraining` | continued pretraining of the encoder |
| `retexo.llm` | the language-model baselines |
