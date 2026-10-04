# Retexo review app

Browse the edit script that retexo predicts for each of the 1,490 references of the Loci Similes benchmark: which reuse
word comes from which source word, and by which operation (`COPY`, `INFLECT`, `SUBST`, `SPLIT`, `MERGE`; `FRAME`, `INS`
and `DEL` for words without a link). Published with the [documentation](https://julianschelb.github.io/retexo/) at [julianschelb.github.io/retexo/demo](https://julianschelb.github.io/retexo/demo/).

A static page (Vite, React, Tailwind CSS), built from the Hugging Face datasets.

## What it shows

- **The pair list** (left): all 1,490 pairs, with search, filters (reference type, source and reuse author and work,
  pairs with a predicted `SUBST` or `INFLECT`), sorting (also by least confident prediction) and pages.
- **The pair** (right): the two passages' citations, authors, works and lengths; the mapping from source to reuse words
  as arrows, horizontal or vertical, with a legend of every label; the Latin texts with their English translations; the
  predicted links with their relation and probability.

The selected pair is in the URL (`#p0006`), so a link opens the same pair.

## Pipeline

1. `scripts/download_data.py` downloads the
   [predicted edit scripts](https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts) and
   the [Loci Similes labels](https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-labels) into
   `data/`.
2. `scripts/prepare_data.py` joins them on the reference's id and writes `public/data/records.jsonl`, one pair per line.
3. `npm run build` bundles the page into `dist/`.
4. `.github/workflows/pages.yml` at the repository root runs all steps on every push to `main` and publishes `dist/`
   under `/demo/` next to the documentation.

## Local development

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
npm install
npm run dev        # http://localhost:5173/retexo/demo/
```

## Layout

| Path | What it holds |
| --- | --- |
| `index.html`, `src/` | the app: `src/App.jsx`, `src/components/`, `src/data.js` (loads the input), `src/filters.js`, `src/labels.js` |
| `scripts/` | the data step: download from the Hub, build `records.jsonl` |
