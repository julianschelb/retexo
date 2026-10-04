# Demo

The [interactive demo](https://julianschelb.github.io/retexo/demo/) shows the predicted edit script of every reference of the Loci Similes benchmark.
The earlier passage is on top and the later passage below; a curve runs from every later word to the source word it takes
up, coloured by label. Hover over a word to see its label, the substitution relation if the model named one, and the
model's confidence. The page also holds the benchmark's reference graph and a browser for its documents.

## How it is built

The demo is a static page, built with React, Vite and Tailwind CSS from the `webapp/` folder, and published with this
documentation by GitHub Actions on every push to `main`.

1. `webapp/scripts/download_data.py` downloads the benchmark (corpus, queries, labels) and the
   [predicted edit scripts](data-format.md) from the Hugging Face Hub.
2. `webapp/scripts/prepare_data.py` turns them into the JSON files the page loads.
3. `npm run build` bundles the page, and the workflow places it under `/demo/` next to the documentation.

The page therefore always shows the published predictions.

## Running it locally

```bash
cd webapp
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
npm install
npm run dev        # http://localhost:5173/retexo/demo/
```
