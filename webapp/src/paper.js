// Static metadata shown in the page header.
export const paper = {
  title: "Learning Word-Level Explanations of Text Reuse in Latin Literature with Little Annotation",
  venue: "Manuscript in preparation",
  authors: [
    { name: "Julian Schelb", affiliation: 1, url: "https://julian-schelb.com" },
    { name: "Michael Wittweiler", affiliation: 3 },
    { name: "Marie Revellio", affiliation: 2 },
  ],
  affiliations: {
    1: "Department of Computer and Information Science, University of Konstanz",
    2: "Department of Latin Philology, University of Konstanz",
    3: "Institute of Archaeology, Classical Philology and Ancient Studies, University of Zurich",
  },
  abstract:
    "Detecting an intertextual reference is only the first step; philologists also need to know how a later author reworked the source. We explain a reuse pair word by word: each word of the reusing passage is first aligned to its source word, then labeled with the operation that transformed it, such as a copy, a change of inflection, or a substitution. Such word-level operations are expensive to annotate, and no corpus annotated at this level exists for text reuse in Latin literature. We describe a procedure that needs only little annotation: synthetic pairs, built by applying known operations to Latin text, train a first model, and active learning chooses the pairs an annotator corrects. Among several architectures, a joint model performs best: a pointer network that, for every reuse word, chooses a source word or none and names the operation in the same step. We show that with 200 pairs chosen by uncertainty, the model matches one trained on every annotated pair of the 1,490 references of the Loci Similes benchmark.",
  // A link left empty is not shown. The paper and its BibTeX follow with the preprint.
  links: {
    pdf: null,
    docs: "https://julianschelb.github.io/retexo/",
    data: "https://huggingface.co/datasets/julian-schelb/latin-classical-intertextuality-edit-scripts",
    code: "https://github.com/julianschelb/retexo",
  },
  bibtex: null,
};
