# Labels

Every link between a reuse word and a source word carries one label. Words without a link are insertions, deletions, or
part of a citing formula.

| Label | Meaning | Example (source / reuse) |
|---|---|---|
| `COPY` | the same form, with spelling variants folded | *frigidus* / *frigidus* |
| `INFLECT` | the same lemma in another form | *regem* / *reges* |
| `SUBST` | another lemma in the same slot | *pugnam* / *calamitatem* |
| `SPLIT` | two reuse words for one source word, usually an enclitic | |
| `MERGE` | one reuse word for two source words | |
| `INS` | a reuse word without a source | |
| `FRAME` | a word of a citing formula such as *ut ait Maro* | |
| `DEL` | a source word that no link claims | |

`INS` and `DEL` follow from the links and are not stored; in the released records they are the `insertions` and
`deletions` lists.

## Substitution relations

A `SUBST` link can name the relation between the two words: `SYN` (same sense, other lemma), `POS` (a derivational
relative), `FUNC` (a function-word swap), `META` (a figurative or metonymic replacement), `NE-SUB` (another proper
name), and others. Where no relation can be named, `relation` is `null`, which is true of a third of the substitutions
in the annotation.

## Levels

The paper scores at a coarser level, **L3**, which folds `SPLIT` and `MERGE` into `SUBST` and `FRAME` into `INS`,
because those three are rare and a macro average weights a rare label as much as a frequent one. The released records
use the fine level, **L4**.

## Conventions

- Spelling is folded before forms are compared (case, u/v, i/j, ae/e, an attached *-que*), so a word that differs only
  there is a `COPY`.
- The lemma decides between inflection and substitution, so a derivational relative counts as a substitution.
- A source word is linked twice only in a split, and that is almost always an enclitic split off its word.
