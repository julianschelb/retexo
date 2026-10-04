# retexo/baselines/llm.py
"""Note 23: the two language-model rows, prompt-only and fine-tuned.

Both write the whole edit script as text in the link-bearing line format
``later_word <- earlier_word  OP`` (one line per reuse word), so a reply names
the source token and the rows produce links *and* types; both are scored on
their own output and are exempt from the shared decoder (``emits = "edges"``,
``decoder = "none"``, ``typer = "own"``).

- ``LLMPromptOnly`` (``llm_prompt``): an instruction model (Claude Sonnet by
  default) with the reuse-gated prompt P4 of ``retexo.llm.script_prompts``,
  supervision "none (prompt)". Replies are written to ``replies.jsonl`` under
  the run directory and the run resumes from it by record id; the cost tally
  stops the run at ``budget_usd``.
- ``LLMQLoRA`` (``llm_qlora``): a QLoRA Llama-3.1-8B-Instruct fine-tuned on the
  training folds' gold scripts (E40's recipe: NF4, rank 16, two epochs, loss
  on the completion only), supervision "gold scripts". Its ``ScriptRater`` is
  the rater of note 16 as well.

``ScriptFormat`` holds the line format (the parser ported verbatim from
``attic/scripts/run_e32_script.parse_words``, the gold script writer, the
annotated passage); ``ScriptParser`` turns a reply into a ``Prediction``.

    python run_baseline.py --method llm_prompt --fold 4 --extra model=claude-sonnet-5,prompt=P4
    python run_baseline.py --method llm_qlora --fold 4
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from retexo.baselines import BaselineRegistry, labels
from retexo.baselines.base import Baseline, BaselineConfig, Prediction
from retexo.baselines.record import Record

#: Operation words the parser accepts on a script line, upper-cased.
OPS_OK = ("NOP", "MORPH", "SUBST", "INS", "FRAME", "SYN-DIST", "SYN", "HYPER",
          "HYPO", "ANT", "NE-SUB", "POS", "SPLIT", "MERGE")

#: Dollars per million input and output tokens (``attic/scripts/generate_scripts.py:139``).
PRICE = {"claude-sonnet-5": (3.0, 15.0), "claude-opus-5": (15.0, 75.0), "claude-haiku-4-5-20251001": (1.0, 5.0)}

#: The prompt-only row's dials and their published defaults.
PROMPT_DEFAULTS: Dict[str, Any] = {"model": "claude-sonnet-5", "prompt": "P4", "effort": "high", "max_tokens": 8000,
                                   "workers": 16, "annotate": "inline", "budget_usd": 30.0, "links_only": 0}

#: The gold's five script operations from a link tag: NOP, MORPH, everything lexical SUBST.
COARSE_SCRIPT_OP = {"NOP": "NOP", "COPY": "NOP", "MORPH": "MORPH"}

#: The fine-tuned row's dials and E40's values.
QLORA_DEFAULTS: Dict[str, Any] = {"model": "meta-llama/Llama-3.1-8B-Instruct", "rank": 16, "lr": 2e-4, "epochs": 2.0,
                                  "grad_accum": 8, "max_len": 3072, "annotate": "inline", "inventory": "coarse",
                                  "load_adapter": "", "dev_pairs": 100}


# =============================================================================
# The line format
# =============================================================================


class ScriptFormat:
    """The word-form script: writing it from gold, reading it from a reply.

    Example:
        ```python
        links, ops, seen = ScriptFormat.parse_words("uitam <- vita,  MORPH\\nsapienti <- -  INS", ["vita,"], ["uitam", "sapienti"])
        # ([0, -1], ["MORPH", "INS"], True)
        ```
    """

    @staticmethod
    def key(word: str) -> str:
        """Case, punctuation, u/v and i/j folded away: how words are matched."""
        return re.sub(r"[^0-9a-z]", "", word.lower().replace("u", "v").replace("j", "i"))

    @classmethod
    def parse_words(cls, text: str, source_tokens: Sequence[str], target_tokens: Sequence[str]
                    ) -> Tuple[List[int], Optional[List[str]], bool]:
        """The model names the words; we do the counting. Each side is matched to
        the next unused position holding that word, so a repeated word goes to
        its next occurrence rather than always the first. Returns ``(links, ops,
        seen)``; ``ops`` is ``None`` and ``seen`` False when no line parsed."""
        src_free: Dict[str, List[int]] = {}
        for i, w in enumerate(source_tokens):
            src_free.setdefault(cls.key(w), []).append(i)
        tgt_pos: Dict[str, List[int]] = {}
        for i, w in enumerate(target_tokens):
            tgt_pos.setdefault(cls.key(w), []).append(i)
        links = [-1] * len(target_tokens)
        ops = ["INS"] * len(target_tokens)
        used_src, used_tgt, seen = set(), set(), False
        for line in text.splitlines():
            line = line.strip().lstrip("-* \t")
            if "<-" not in line:
                continue
            left, right = line.split("<-", 1)
            cands = [i for i in tgt_pos.get(cls.key(left), []) if i not in used_tgt]
            if not cands:
                continue
            t = cands[0]
            used_tgt.add(t)
            seen = True
            parts = right.split()
            op = next((p.upper() for p in parts if p.upper() in OPS_OK), None)
            if op:
                ops[t] = op
            rest = [p for p in parts if p.upper() not in OPS_OK]
            if rest and rest[0].strip() not in ("-", "--", "none", "None"):
                free = [i for i in src_free.get(cls.key(rest[0]), []) if i not in used_src]
                if free:
                    links[t] = free[0]
                    used_src.add(free[0])
        return links, (ops if seen else None), seen

    @staticmethod
    def gold_script(source_tokens: Sequence[str], target_tokens: Sequence[str], links: Sequence[int],
                    ops: Sequence[str]) -> str:
        """One line per reuse word, the gold's own operation words."""
        lines = []
        for t, (op, s) in enumerate(zip(ops, links)):
            src = source_tokens[s] if s >= 0 else "-"
            lines.append(f"{target_tokens[t]} <- {src}  {op}")
        return "\n".join(lines)

    @staticmethod
    def plain(tokens: Sequence[str]) -> str:
        """The passage with word indices and no dictionary entries (no resources)."""
        return "\n".join(f"{i:>3}: {w}" for i, w in enumerate(tokens))

    @staticmethod
    def annotated(tokens: Sequence[str], featurizer, resources, *, neighbours: bool = True) -> str:
        """The passage itself, each word carrying its dictionary entry inline."""
        lines = []
        for i, w in enumerate(tokens):
            lemma = featurizer.lemma(w) or "?"
            try:
                pos = resources.pos_of(lemma) or ""
            except Exception:
                pos = ""
            bits = [lemma] if lemma != "?" else ["not in the dictionary"]
            if pos:
                bits.append({"n": "noun", "v": "verb", "a": "adjective", "r": "adverb"}.get(pos, pos))
            try:
                if resources.entities and resources.entities.is_name(w):
                    bits.append("proper name")
            except Exception:
                pass
            if neighbours and lemma != "?":
                try:
                    rec = resources.wordnet.lookup(lemma, pos or "n")
                except Exception:
                    rec = {}
                for field, name in (("synonyms", "= "), ("hypernyms", "a kind of "), ("antonyms", "opposite of ")):
                    got = (rec.get(field) or [])[:3]
                    if got:
                        bits.append(name + ", ".join(got))
            lines.append(f"{i:>3}: {w:<16} [{'; '.join(bits)}]")
        return "\n".join(lines)


class PassageWriter:
    """Renders a passage for a prompt: annotated through the Latin resources,
    or plain when ``annotate`` is ``none`` (an English set, or no resources).

    Example:
        ```python
        writer = PassageWriter("inline")
        text = writer(record.source_tokens)
        ```
    """

    def __init__(self, annotate: str = "inline"):
        self.annotate = annotate
        self._featurizer = None
        self._resources = None

    def _load(self) -> bool:
        if self._featurizer is not None:
            return True
        try:
            from retexo.edit_typing.link_features import LinkFeaturizer
            from retexo.resources import Resources

            self._resources = Resources(offline=True)
            self._featurizer = LinkFeaturizer(self._resources)
            return True
        except Exception:
            return False

    def __call__(self, tokens: Sequence[str]) -> str:
        if self.annotate == "inline" and self._load():
            return ScriptFormat.annotated(tokens, self._featurizer, self._resources)
        return ScriptFormat.plain(tokens)


# =============================================================================
# Reply to prediction
# =============================================================================


class ScriptParser:
    """A reply in the line format to a ``Prediction``.

    Example:
        ```python
        pred = ScriptParser.to_prediction(reply, record)
        pred.meta["parsed"]      # False when no line carried "<-": pred.invalid is then True
        ```
    """

    @staticmethod
    def to_prediction(text: str, record: Record, *, links_only: bool = False) -> Prediction:
        links, ops, seen = ScriptFormat.parse_words(text, record.source_tokens, record.reuse_tokens)
        if not seen:
            pred = Prediction(links=[], tags=[], frame=[], raw=text)
            pred.meta["parsed"] = False
            return pred
        n = record.n_reuse
        tags, frame = [""] * n, [0] * n
        for t in range(n):
            op = ops[t] if ops else "INS"
            if links[t] >= 0:
                canon, detail = labels.canonical(op)
                tags[t] = "SUBST" if links_only and canon not in ("COPY", "MORPH") else (canon or "SUBST")
            elif op == "FRAME":
                frame[t] = 1
        pred = Prediction(links=list(links), tags=tags, frame=frame)
        pred.meta["parsed"] = True
        pred.meta["ops"] = ops
        return pred


# =============================================================================
# The prompt-only row
# =============================================================================


@dataclass
class CostTally:
    """Token counts and their price for one run.

    Example:
        ```python
        tally = CostTally("claude-sonnet-5")
        tally.add(usage); tally.dollars()
        ```
    """

    model: str
    n: int = 0
    parsed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def add(self, usage, *, parsed: bool) -> None:
        self.n += 1
        self.parsed += int(parsed)
        self.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
        self.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_read += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        self.cache_write += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    def dollars(self) -> float:
        pi, po = PRICE.get(self.model, (3.0, 15.0))
        return (self.tokens_in / 1e6 * pi + self.tokens_out / 1e6 * po
                + self.cache_read / 1e6 * pi * 0.1 + self.cache_write / 1e6 * pi * 1.25)

    def as_dict(self) -> Dict[str, Any]:
        return {**self.__dict__, "usd": round(self.dollars(), 4)}


@BaselineRegistry.register
class LLMPromptOnly(Baseline):
    """"LLM, prompt only": the instruction model writes the script from the
    prompt alone. ``cfg.extra``: ``model``, ``prompt`` (P0 to P5), ``effort``,
    ``max_tokens``, ``workers``, ``annotate`` (``inline`` | ``none``),
    ``budget_usd``, ``links_only`` (types stripped to COPY / MORPH / SUBST for
    the LLM-labelled level of note 33).

    Example:
        ```python
        method = LLMPromptOnly(BaselineConfig(out=Path("runs/llm_p4"), extra={"budget_usd": 5}))
        preds = method.predict(records)            # resumes from runs/llm_p4/replies.jsonl
        # python run_baseline.py --method llm_prompt --fold 4 --extra prompt=P4
        ```
    """

    name = "llm_prompt"
    emits = "edges"
    trainable = False
    typer = "own"
    decoder = "none"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**PROMPT_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in PROMPT_DEFAULTS}}
        self.writer = PassageWriter(str(self.dials["annotate"]))
        self.tally = CostTally(str(self.dials["model"]))
        self._client = None

    # ---------- prompts ----------

    def prompt_for(self, record: Record) -> Tuple[str, str]:
        """``(system text, user text)`` of the configured variant for one pair."""
        from retexo.llm.script_prompts import PromptVariants

        system, template = PromptVariants.get(str(self.dials["prompt"]))
        n = record.n_reuse
        user = template.format(source_annotated=self.writer(record.source_tokens),
                               target_annotated=self.writer(record.reuse_tokens),
                               n_lines_note=f"That is {n} lines, one per word, in order. ")
        return system, user

    # ---------- the API ----------

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(default_headers={"Accept-Encoding": "gzip"}, timeout=120, max_retries=2)
        return self._client

    def ask(self, system: str, user: str) -> Tuple[Optional[str], Any]:
        """One reply; adaptive thinking first, a plain second attempt when the
        model spent its budget thinking and said nothing. Returns ``(text, usage)``."""
        blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        plans = [dict(thinking={"type": "adaptive"}, output_config={"effort": str(self.dials["effort"])},
                      max_tokens=int(self.dials["max_tokens"])),
                 dict(thinking={"type": "disabled"}, max_tokens=1200)]
        for plan in plans:
            response = None
            for attempt in range(3):
                try:
                    response = self.client().messages.create(model=str(self.dials["model"]), system=blocks,
                                                             messages=[{"role": "user", "content": user}], **plan)
                    break
                except TypeError:
                    raise
                except Exception:
                    time.sleep(5 * (attempt + 1))
            if response is None:
                continue
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            if text.strip():
                return text, response.usage
        return None, None

    # ---------- inference ----------

    def replies_path(self) -> Path:
        return Path(self.cfg.out) / "replies.jsonl"

    def stored_replies(self) -> Dict[str, str]:
        path = self.replies_path()
        if not path.exists():
            return {}
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row["reply"]
        return out

    def predict(self, records: List[Record]) -> List[Prediction]:
        from concurrent.futures import ThreadPoolExecutor

        replies = self.stored_replies()
        todo = [r for r in records if r.id not in replies]
        path = self.replies_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        budget = float(self.dials["budget_usd"])

        def one(record: Record) -> Optional[str]:
            if self.tally.dollars() > budget:
                return None
            system, user = self.prompt_for(record)
            text, usage = self.ask(system, user)
            if text is None:
                return None
            _, _, seen = ScriptFormat.parse_words(text, record.source_tokens, record.reuse_tokens)
            self.tally.add(usage, parsed=seen)
            with path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"id": record.id, "reply": text, "parsed": seen, "model": self.dials["model"],
                                       "prompt": self.dials["prompt"]}, ensure_ascii=False) + "\n")
            return text

        if todo:
            with ThreadPoolExecutor(max_workers=max(1, int(self.dials["workers"]))) as pool:
                for record, text in zip(todo, pool.map(one, todo)):
                    if text is not None:
                        replies[record.id] = text
        links_only = bool(int(self.dials["links_only"]))
        out = []
        for record in records:
            text = replies.get(record.id)
            if text is None:
                pred = Prediction(links=[], tags=[], frame=[], raw="")
                pred.meta["parsed"] = False
                out.append(pred)
            else:
                out.append(ScriptParser.to_prediction(text, record, links_only=links_only))
        return out

    def label_records(self, records: List[Record]) -> List[Record]:
        """The records with the model's edges as their ``pred`` (note 33's LLM-labelled level)."""
        from dataclasses import replace

        from retexo.baselines.record import RecordInterface

        out = []
        for record, pred in zip(records, self.predict(records)):
            edges = [] if pred.invalid else RecordInterface.edges_from(pred.links, pred.tags, pred.frame, pred.extra)
            out.append(replace(record, pred={"edges": edges, "parsed": pred.meta.get("parsed", False)}))
        return out


# =============================================================================
# The fine-tuned row and its rater
# =============================================================================


class _AdapterState:
    """The LoRA adapter's weights alone, with ``state_dict`` / ``load_state_dict``, so early stopping
    copies megabytes (the adapter) instead of the quantised base model."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.model)

    def load_state_dict(self, state) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)


class ScriptRater:
    """A QLoRA-tuned instruction model that writes the script (E40's recipe),
    shared by ``LLMQLoRA`` and the full system's rater.

    Example:
        ```python
        rater = ScriptRater(QLORA_DEFAULTS, device="cuda", annotate="none")
        rater.train(train_records, Path("runs/x/rater"), log=print)
        replies = rater.generate(test_records)            # one text per record
        links = [ScriptFormat.parse_words(r, rec.source_tokens, rec.reuse_tokens)[0] for r, rec in zip(replies, test_records)]
        ```
    """

    def __init__(self, dials: Dict[str, Any], *, device: str = "cuda", annotate: str = "inline"):
        self.dials = {**QLORA_DEFAULTS, **dials}
        self.device = device
        self.writer = PassageWriter(annotate)
        self.model = None
        self.tokenizer = None

    # ---------- prompts ----------

    def messages(self, record: Record) -> List[Dict[str, str]]:
        from retexo.llm.script_prompts import RATER_SYSTEM, RATER_USER

        user = RATER_USER.format(src=self.writer(record.source_tokens), tgt=self.writer(record.reuse_tokens),
                                 n=record.n_reuse)
        return [{"role": "system", "content": RATER_SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def completion(record: Record, inventory: str = "coarse") -> str:
        """The gold script of a record in the rater's operation words."""
        from retexo.baselines.record import RecordInterface

        links, tags, frame, _ = RecordInterface.links_of(record)
        ops = []
        for t in range(record.n_reuse):
            if links[t] < 0:
                ops.append("FRAME" if frame[t] else "INS")
            else:
                op = labels.TO_LINK_TAG.get(tags[t], tags[t])
                ops.append(op if inventory == "fine" else COARSE_SCRIPT_OP.get(op, "SUBST"))
        return ScriptFormat.gold_script(record.source_tokens, record.reuse_tokens, links, ops)

    # ---------- the model ----------

    def load_base(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        name = str(self.dials["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map={"": 0},
                                                          dtype=torch.bfloat16)
        return self.model

    def encode(self, messages, completion: str):
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tokenizer(completion + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        max_len = int(self.dials["max_len"])
        ids = (p_ids + c_ids)[:max_len]
        labels_ = ([-100] * len(p_ids) + c_ids)[:max_len]
        return torch.tensor(ids), torch.tensor(labels_)

    def train(self, records: Sequence[Record], out_dir: Path, *, log=None, stopper_factory=None,
              valid: Sequence[Record] = ()) -> Path:
        """QLoRA on the records' gold scripts; the adapter is saved under ``out_dir/adapter``.

        With ``valid`` scripts and a ``stopper_factory`` (a callable taking the loss function and
        returning an ``early_stopping.EarlyStopping``), training runs whole epochs up to the
        stopper's maximum, scores the validation scripts' completion loss after each, and keeps
        the best epoch's adapter weights."""
        import math
        import random

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import get_cosine_schedule_with_warmup

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model = self.load_base()
        data = [(self.messages(r), self.completion(r, str(self.dials["inventory"]))) for r in records]
        random.Random(1).shuffle(data)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        rank = int(self.dials["rank"])
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=float(self.dials["lr"]), weight_decay=0.0)
        epochs, accum = float(self.dials["epochs"]), int(self.dials["grad_accum"])
        steps = math.ceil(len(data) * epochs / accum)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * steps), steps)
        encoded = [self.encode(m, c) for m, c in data]
        encoded = [(i, l) for i, l in encoded if int((l != -100).sum()) > 0]      # a fully masked sample gives 0/0
        valid_encoded = [(i, l) for i, l in (self.encode(self.messages(r), self.completion(r, str(self.dials["inventory"])))
                                             for r in valid) if int((l != -100).sum()) > 0]

        def valid_loss() -> float:
            model.eval()
            with torch.no_grad():
                losses = [float(model(input_ids=i[None].to(self.device), labels=l[None].to(self.device)).loss)
                          for i, l in valid_encoded]
            model.train()
            return sum(losses) / max(len(losses), 1)

        stopper = stopper_factory(valid_loss) if (stopper_factory is not None and valid_encoded) else None
        if stopper is not None:
            epochs = float(stopper.max_epochs)
            steps = math.ceil(len(encoded) * epochs / accum)
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * steps), steps)
        adapter = _AdapterState(model)
        n_epochs = int(math.ceil(epochs))
        model.train()
        started, step, seen, running = time.time(), 0, 0, 0.0
        for epoch in range(1, n_epochs + 1):
            share = min(1.0, epochs - (epoch - 1))                # a fractional last epoch (e.g. 1.5) runs part of the data
            for i in list(range(len(encoded)))[: int(len(encoded) * share)]:
                ids, target = encoded[i]
                output = model(input_ids=ids[None].to(self.device), labels=target[None].to(self.device))
                if not torch.isfinite(output.loss):
                    optimizer.zero_grad()
                    continue
                (output.loss / accum).backward()
                running += float(output.loss.item()); seen += 1
                if seen % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(); step += 1
                    if log and step % 10 == 0:
                        log(f"[rater] step {step}/{steps} loss {running / seen:.3f} {(time.time() - started) / 60:.1f} min")
                        running, seen = 0.0, 0
            if stopper is not None and not stopper.step(epoch, {"adapter": adapter}):
                break
        if stopper is not None:
            stopper.restore({"adapter": adapter})
            stopper.release()
        model.save_pretrained(out_dir / "adapter")
        self.model = model
        if log:
            log(f"[rater] trained on {len(encoded)} scripts in {(time.time() - started) / 60:.1f} min -> {out_dir / 'adapter'}")
        return out_dir / "adapter"

    def load_adapter(self, adapter_dir: Path):
        from peft import PeftModel

        base = self.load_base()
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        return self.model

    def generate(self, records: Sequence[Record], *, log=None) -> List[str]:
        """One reply per record, greedy, ``max_new_tokens = min(12 n + 40, 1800)``."""
        import torch

        if self.model is None:
            raise RuntimeError("the rater has no adapter: train() or load_adapter() first")
        self.model.eval()
        self.tokenizer.padding_side = "left"
        out, started = [], time.time()
        for i, record in enumerate(records):
            prompt = self.tokenizer.apply_chat_template(self.messages(record), tokenize=False, add_generation_prompt=True)
            batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(**batch, max_new_tokens=min(12 * record.n_reuse + 40, 1800),
                                                do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            out.append(self.tokenizer.decode(generated[0][batch["input_ids"].shape[1]:], skip_special_tokens=True))
            if log and (i + 1) % 25 == 0:
                log(f"[rater] generated {i + 1}/{len(records)} {(time.time() - started) / 60:.1f} min")
        return out


@BaselineRegistry.register
class LLMQLoRA(Baseline):
    """"LLM, fine-tuned": the QLoRA script writer trained on the training folds'
    gold scripts. ``cfg.extra``: ``model``, ``rank``, ``lr``, ``epochs``,
    ``grad_accum``, ``max_len``, ``annotate``, ``inventory`` (``coarse`` |
    ``fine``), ``load_adapter``, ``dev_pairs``.

    Example:
        ```python
        method = LLMQLoRA(cfg).fit(train, dev, log=print)
        preds = method.predict(test)
        # python run_baseline.py --method llm_qlora --fold 4 --save-model
        ```
    """

    name = "llm_qlora"
    early_stopping_capable = True
    emits = "edges"
    trainable = True
    typer = "own"
    decoder = "none"

    def __init__(self, cfg: BaselineConfig):
        super().__init__(cfg)
        self.dials = {**QLORA_DEFAULTS, **{k: v for k, v in cfg.extra.items() if k in QLORA_DEFAULTS}}
        self.rater = ScriptRater(self.dials, device=cfg.device, annotate=str(self.dials["annotate"]))
        self.replies: Dict[str, str] = {}

    def fit(self, train: List[Record], dev: List[Record], *, log=None, unlabeled: Sequence[Record] = ()) -> "LLMQLoRA":
        adapter = str(self.dials["load_adapter"])
        if adapter:
            self.rater.load_adapter(Path(adapter))
            return self
        if self.cfg.smoke:
            self.dials["epochs"] = min(float(self.dials["epochs"]), 1.0)
            self.rater.dials["epochs"] = self.dials["epochs"]
        from retexo.baselines.early_stopping import EarlyStopping

        factory = (lambda loss: EarlyStopping.for_method(self, log=log, score=loss, higher_is_better=False,
                                                         metric="validation completion loss")) if self.validation else None
        # the annotated scripts and the shared negatives (every line INS); the synthetic pairs are left out,
        # since 25,000 full prompts through an 8B model would take days
        scripts = list(train) + list(self.shared_negatives[: len(train)])
        self.rater.train(scripts, Path(self.cfg.out) / "rater", log=log, stopper_factory=factory, valid=self.validation)
        if dev and log:
            sample = dev[: int(self.dials["dev_pairs"])]
            replies = self.rater.generate(sample)
            parsed = sum(ScriptFormat.parse_words(r, rec.source_tokens, rec.reuse_tokens)[2] for r, rec in zip(replies, sample))
            log(f"[llm_qlora] dev parse rate {parsed}/{len(sample)}")
        return self

    def predict(self, records: List[Record]) -> List[Prediction]:
        todo = [r for r in records if r.id not in self.replies]
        if todo:
            if self.rater.model is None:
                return [Prediction(links=[], tags=[], frame=[], raw="") for _ in records]
            for record, text in zip(todo, self.rater.generate(todo)):
                self.replies[record.id] = text
        path = Path(self.cfg.out) / "replies.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as sink:
            for record in records:
                sink.write(json.dumps({"id": record.id, "reply": self.replies[record.id]}, ensure_ascii=False) + "\n")
        return [ScriptParser.to_prediction(self.replies[r.id], r) for r in records]

    def save(self, path: Path) -> None:
        """The adapter directory (the base weights are the hub's)."""
        if self.rater.model is not None:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.rater.model.save_pretrained(str(Path(path) / "adapter"))

    @classmethod
    def load(cls, path: Path, cfg: BaselineConfig) -> "LLMQLoRA":
        method = cls(cfg)
        method.rater.load_adapter(Path(path) / "adapter")
        return method
