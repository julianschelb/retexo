// Display helpers shared by the list and the detail pane.

// Operation colours; class names are spelled out so Tailwind finds them.
export const OP_STYLE = {
  COPY: { text: "text-op-copy", bg: "bg-op-copy-soft", border: "border-op-copy", outline: "outline-op-copy" },
  INFLECT: { text: "text-op-inflect", bg: "bg-op-inflect-soft", border: "border-op-inflect", outline: "outline-op-inflect" },
  SUBST: { text: "text-op-subst", bg: "bg-op-subst-soft", border: "border-op-subst", outline: "outline-op-subst" },
  SPLIT: { text: "text-op-split", bg: "bg-op-split-soft", border: "border-op-split", outline: "outline-op-split" },
  MERGE: { text: "text-op-split", bg: "bg-op-split-soft", border: "border-op-split", outline: "outline-op-split" },
  FRAME: { text: "text-muted", bg: "bg-line-soft", border: "border-muted", outline: "outline-muted" },
  INS: { text: "text-ink", bg: "", border: "border-transparent", outline: "outline-muted" },
};

export const LABEL_NAME = { cit: "cit.", cf: "cf.", no_match: "no match" };
