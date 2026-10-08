"""Smoke test: does the generated data actually feed a model and train?

Uses deliberately tiny random-weight models. The point is not accuracy — with
121k random parameters there is none — but that the pipeline runs end to end:
generation, canonical scripts, encoding, tokenisation, a forward pass and a
backward pass that lowers the loss on a batch it has seen repeatedly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from retexo.datasets.generation import (  # noqa: E402
    GenerationConfig,
    MockSubstitutionSource,
    SyntheticGenerator,
)
from retexo.formulations.encoding import DERIVED_TAGS, ScriptEncoder  # noqa: E402
from retexo.operations import OperationRegistry  # noqa: E402

SEEDS = [
    "arma uirumque cano Troiae qui primus ab oris",
    "uox faucibus haesit et inter ruborem atque pallorem",
    "ingentes animos angusto in pectore uersant",
    "obstipui steteruntque comae et uox faucibus haesit",
    "at regina graui iamdudum saucia cura uulnus alit",
]


def build_dataset(n_per_seed: int = 12):
    generator = SyntheticGenerator(
        MockSubstitutionSource(), GenerationConfig(variants_per_seed=n_per_seed, seed=11)
    )
    examples, report = generator.generate(SEEDS)
    print(f"[data] {report.summary()}")
    return examples


def smoke_seq2seq(examples) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "hf-internal-testing/tiny-random-t5"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)

    added = tokenizer.add_tokens(ScriptEncoder.operation_tokens(OperationRegistry.default().tags()))
    model.resize_token_embeddings(len(tokenizer))
    print(
        f"[seq2seq] {sum(p.numel() for p in model.parameters()):,} params, "
        f"{added} operation tokens added"
    )

    pairs = [ScriptEncoder.to_seq2seq(e["script"]) for e in examples[:16]]
    batch = tokenizer(
        [p[0] for p in pairs], padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    labels = tokenizer(
        [p[1] for p in pairs], padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = last = None
    for step in range(30):
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"[seq2seq] loss {first:.4f} -> {last:.4f}")
    assert last < first, "seq2seq loss did not decrease"


def smoke_token_classifier(examples) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    name = "hf-internal-testing/tiny-random-bert"
    # Mirrors TokenClassifierModel: insertion and deletion are read off the
    # alignment, not predicted, so they are absent from the operation head.
    vocabulary = ScriptEncoder.label_vocabulary(
        [t for t in OperationRegistry.default().tags() if t not in DERIVED_TAGS]
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForTokenClassification.from_pretrained(
        name, num_labels=len(vocabulary), ignore_mismatched_sizes=True
    )
    print(
        f"[token-cls] {sum(p.numel() for p in model.parameters()):,} params, "
        f"{len(vocabulary)} labels"
    )

    encoded = [ScriptEncoder.to_token_labels(e["script"]) for e in examples[:16]]
    batch = tokenizer(
        [e["tokens"] for e in encoded],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    labels = torch.full(batch.input_ids.shape, -100)
    for row, item in enumerate(encoded):
        word_ids = batch.word_ids(row)
        seen = set()
        for position, word in enumerate(word_ids):
            if word is None or word in seen:
                continue
            seen.add(word)
            tag = item["op_labels"][word]
            if tag in vocabulary:  # insertions carry no operation label
                labels[row, position] = vocabulary[tag]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = last = None
    for step in range(30):
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"[token-cls] loss {first:.4f} -> {last:.4f}")
    assert last < first, "token-classifier loss did not decrease"


if __name__ == "__main__":
    torch.manual_seed(0)
    data = build_dataset()
    smoke_seq2seq(data)
    smoke_token_classifier(data)
    print("\nsmoke test passed")
