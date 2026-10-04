# Retexo demo page

Read-only web demo of retexo's predicted edit scripts: the script of every reference of the Loci Similes benchmark,
drawn as curves between the two passages, plus the benchmark's reference graph and a document browser. Built with
React, Vite and Tailwind CSS, and published with the documentation at https://julianschelb.github.io/retexo/demo/.

## Pipeline

1. `scripts/download_data.py` downloads the benchmark (`corpus`, `queries`, `labels`) and the predicted edit scripts
   (`edit_scripts`) from the Hugging Face Hub into `data/`.
2. `scripts/prepare_data.py` converts them into JSON under `public/data/` for the page.
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

- `src/paper.js`: metadata shown in the header (title, authors, abstract, links).
- `src/components/EditScripts.jsx`: the edit-script viewer.
- `src/components/ReferenceGraph.jsx`, `DocumentBrowser.jsx`, `FilterBar.jsx`: the benchmark views.
- `scripts/prepare_data.py`: `build_scripts` turns the released records into the viewer's payload.
