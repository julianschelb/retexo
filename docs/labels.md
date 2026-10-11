# Labels

Every link between a reuse word and a source word carries one label. A word without a link is part of a citing formula,
or an insertion or deletion inside the reused stretch, or new text or text not taken up outside it. The reused stretch of
a passage runs from its first to its last linked word.

| Label | Meaning | Example (source / reuse) |
|---|---|---|
| `COPY` | the same form, with spelling variants folded | *frigidus* / *frigidus* |
| `INFLECT` | the same lemma in another form | *regem* / *reges* |
| `SUBST` | another lemma in the same slot | *pugnam* / *calamitatem* |
| `SPLIT` | two reuse words for one source word, usually an enclitic | |
| `MERGE` | one reuse word for two source words | |
| `MARK` | a word of a citing formula such as *ut ait Maro* | |
| `INS` | a reuse word without a source, inside the reused stretch | |
| `NEW` | a reuse word without a source, outside the reused stretch: the later author's own text | |
| `DEL` | a source word that no link claims, inside the reused stretch | |
| `OUT` | a source word that no link claims, outside the reused stretch: source text not taken up | |

`INS`, `NEW`, `DEL` and `OUT` follow from the links and are not predicted on their own; in the released records they are
the `insertions`, `new_text`, `deletions` and `not_taken_up` lists.

## Patterns

Three patterns are read off a finished script and never predicted on their own: `QUOTE`, a run of at least four copies
in source order; `REORDER`, a reuse word whose link lies outside the longest chain of links in source order (a moved
word); and `DISPERSE`, a clause break inside the linked range of the reusing passage. The released records hold them in
`patterns`.

## Substitution relations

A `SUBST` link can name the relation between the two words: `SYN` (same sense, other lemma), `DERIV` (a derivational
relative), `FUNC` (a function-word swap), `META` (a figurative or metonymic replacement), `NE-SUB` (another proper
name), and others. Where no relation can be named, `relation` is `null`, which is true of a third of the substitutions
in the annotation.

## Levels

The paper scores at a coarser level, **L3**, which folds `SPLIT` and `MERGE` into `SUBST`, `MARK` and `NEW` into `INS`,
and `OUT` into `DEL`: splits, merges and citing formulas are rare, and a macro average weights a rare label as much as a
frequent one. The released records use the fine level, **L4**.

Until 2026-10-12 the citing formula was called `FRAME`, the derivational relative `POS`, and `INS` and `DEL` covered
every word without a link.

## Conventions

- Spelling is folded before forms are compared (case, u/v, i/j, ae/e, an attached *-que*), so a word that differs only
  there is a `COPY`.
- The lemma decides between inflection and substitution, so a derivational relative counts as a substitution.
- A source word is linked twice only in a split, and that is almost always an enclitic split off its word.
