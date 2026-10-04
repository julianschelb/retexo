# Web Demo

The [review app](https://julianschelb.github.io/retexo/demo/) lets you browse the edit script that retexo predicts for
each of the 1,490 references of the Loci Similes benchmark.

- **The pair list** has search, filters (reference type, source and reuse author and work, pairs with a predicted
  substitution or inflection) and sorting, including the least confident predictions first.
- **The pair** shows the mapping from source to reuse words as arrows, horizontal or vertical, coloured by label and
  thicker the more confident the prediction; the Latin texts with their English translations; and a table of the
  predicted links with their relation and probability.

The selected pair is in the URL, so a link opens the same pair.

## How it is built

The app is a static page, built with React, Vite and Tailwind CSS from the `webapp/` folder, and published with this
documentation by GitHub Actions on every push to `main`.

1. `webapp/scripts/download_data.py` downloads the [predicted edit scripts](data-format.md) and the Loci Similes labels
   from the Hugging Face Hub.
2. `webapp/scripts/prepare_data.py` joins them on the reference's id into `records.jsonl`.
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
