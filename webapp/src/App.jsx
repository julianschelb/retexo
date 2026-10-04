import { useEffect, useState } from "react";
import PairList from "./components/PairList.jsx";
import PairDetail from "./components/PairDetail.jsx";
import { DATA_FILE, loadRecords } from "./data.js";

const idFromHash = () => decodeURIComponent(window.location.hash.slice(1)) || null;

export default function App() {
  const [records, setRecords] = useState(null);
  const [error, setError] = useState(null);
  const [selectedId, setSelectedId] = useState(idFromHash);

  // the input: the annotation file (src/data.js); the first pair is selected unless the URL names one
  useEffect(() => {
    loadRecords()
      .then((loaded) => {
        setRecords(loaded);
        setSelectedId((id) => (id && loaded.some((r) => r.id === id) ? id : loaded[0]?.id ?? null));
      })
      .catch((e) => setError(e.message));
  }, []);

  // the selected pair lives in the URL hash, so a link opens the same pair
  useEffect(() => {
    if (selectedId && idFromHash() !== selectedId) window.history.replaceState(null, "", `#${selectedId}`);
  }, [selectedId]);
  useEffect(() => {
    const onHash = () => setSelectedId(idFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const selected = records?.find((r) => r.id === selectedId) ?? null;

  return (
    <div className="flex h-dvh flex-col">
      <header className="relative z-20 flex items-center gap-3 border-b border-black/10 bg-linear-to-b from-header-top to-header-bottom px-4 py-2.5">
        <h1 className="font-serif text-lg font-semibold">
          <span className="text-pop">Retexo</span> <span className="text-accent">Reuse Edit Operations</span>
        </h1>
        <nav className="ml-auto flex gap-4 text-xs font-semibold" aria-label="Project links">
          <a className="text-accent hover:text-pop" href="https://julianschelb.github.io/retexo/" target="_blank" rel="noopener">Documentation</a>
          <a className="text-accent hover:text-pop" href="https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts" target="_blank" rel="noopener">Predictions</a>
          <a className="text-accent hover:text-pop" href="https://github.com/julianschelb/retexo" target="_blank" rel="noopener">Code</a>
        </nav>
      </header>
      {!records ? (
        <div className="grid flex-1 place-items-center p-8 text-sm text-muted">
          {error ? `Could not load the predictions (${DATA_FILE}): ${error}` : "Loading the predictions…"}
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,40%)_minmax(0,1fr)] md:grid-cols-[29rem_minmax(0,1fr)] md:grid-rows-1">
          <PairList records={records} selectedId={selectedId} onSelect={setSelectedId} />
          <main className="min-h-0 overflow-y-auto">
            <PairDetail record={selected} />
          </main>
        </div>
      )}
    </div>
  );
}
