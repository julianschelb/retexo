import { OP_STYLE } from "../labels.js";

// Every label a word can carry in the mapping, in the paper's operation inventory.
const TAGS = [
  ["COPY", "The same word, taken over unchanged. A spelling variant (temptare / tentare) or an enclitic added or dropped (aristis / aristisque) still counts as a copy."],
  ["INFLECT", "The same word in another form: same lemma, different case, number, tense, mood or person (est / esse)."],
  ["SUBST", "A different word in the same slot: a synonym, a derivation, another name (Danaos / Graecos), or a replacement with no lexical relation."],
  ["SPLIT", "One source word becomes two reuse words, e.g. a glued enclitic written apart (nostroque / nostro, quae). Both reuse words point to the same source word."],
  ["MERGE", "Two source words become one reuse word. The link points to the first of them."],
  ["FRAME", "The citing formula: the later author's own words introducing the quotation (ut ait poeta, inquit). No source word."],
  ["INS", "Inserted: a word of the later author with no counterpart in the source."],
  ["DEL", "Deleted: a source word the later author did not take over (source side only, shown in grey)."],
];

export default function Legend() {
  return (
    <div className="mt-4 border-t border-line pt-3">
      <p className="mb-2 text-[.65rem] font-semibold uppercase tracking-[.08em] text-muted">Legend</p>
      <dl className="grid gap-x-6 gap-y-1.5 text-xs text-ink-2 md:grid-cols-2">
        {TAGS.map(([tag, text]) => {
          const style = OP_STYLE[tag] ?? OP_STYLE.INS;
          const muted = tag === "INS" || tag === "DEL" || tag === "FRAME";
          return (
            <div key={tag} className="flex gap-2">
              <dt className={`w-14 shrink-0 pt-px text-[.65rem] font-semibold uppercase tracking-wide ${muted ? "text-muted" : style.text}`}>{tag}</dt>
              <dd className="leading-snug">{text}</dd>
            </div>
          );
        })}
      </dl>
      <p className="mt-2 text-xs text-muted">
        Arrows run from the source word to the reuse word; a thicker, darker arrow is a more confident prediction.
        Hover a word to see its link. Long passages scroll sideways.
      </p>
    </div>
  );
}
