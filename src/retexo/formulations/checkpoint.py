# retexo/formulations/checkpoint.py
"""Loading a preliminary-round pointer checkpoint (``run_e5.save_model``'s layout).

The E5 to E10 runs saved a detector as ``config.json`` (the frozen
``ChangeDetectorConfig`` plus the class list), ``encoder/`` (the backbone in
Hugging Face format) and ``heads.pt`` (every head's state). ``run_e5.load_model``
rebuilt it; that script lives in the attic and imports modules that no longer
exist, so the loader is ported here verbatim in substance: the config is
rebuilt from the saved JSON rather than re-specified, so a checkpoint cannot
come up under a configuration it was not trained under.

    model, classes = PointerCheckpoint.load(Path("attic/runs/best_models"), device="cuda")
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple


class PointerCheckpoint:
    """A saved ``ChangeDetector`` or ``TypedPointer`` with its class list.

    Example:
        ```python
        model, classes = PointerCheckpoint.load(Path("attic/runs/best_models"))
        rows = model.predict_alignment_scores([example])
        ```
    """

    @staticmethod
    def load(directory: Path, *, device: str = "cuda") -> Tuple[object, List[str]]:
        import torch
        from transformers import AutoModel

        from retexo.formulations.change_detector import ChangeDetector, ChangeDetectorConfig

        directory = Path(directory)
        saved = json.loads((directory / "config.json").read_text())
        classes = saved.pop("classes", list(ChangeDetectorConfig().operations))
        names = {f.name for f in fields(ChangeDetectorConfig)}
        kwargs = {k: v for k, v in saved.items() if k in names}
        kwargs.update(device=device, operations=tuple(classes))
        for key in ("class_weights", "logit_bias", "fine_operations", "fine_class_weights"):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(tuple(v) if isinstance(v, list) else v for v in kwargs[key])
        config = ChangeDetectorConfig(**kwargs)
        if getattr(config, "typed_pointer", False):
            from retexo.formulations.typed_pointer import TypedPointer

            model = TypedPointer(config)
        else:
            model = ChangeDetector(config)
        encoder = AutoModel.from_pretrained(directory / "encoder")
        target = model._encoder.module if hasattr(model._encoder, "module") else model._encoder
        target.load_state_dict(encoder.state_dict())
        heads = torch.load(directory / "heads.pt", map_location=device)
        model._head.load_state_dict(heads["head"])
        if model._source_head is not None and "source_head" in heads:
            model._source_head.load_state_dict(heads["source_head"])
        if config.pointer:
            if "pointer_source" not in heads:
                raise ValueError(f"{directory} was saved before pointer weights were kept; "
                                 "its pointer would reload randomly initialised")
            model._pointer_source.load_state_dict(heads["pointer_source"])
            model._pointer_target.load_state_dict(heads["pointer_target"])
            model._pointer_null.data = heads["pointer_null"].to(device)
            if "pointer_temperature" in heads:
                model._pointer_temperature.data = heads["pointer_temperature"].to(device)
        for name, key in (("_typer", "typer"), ("_typer_evidence", "typer_evidence"), ("_frame_head", "frame_head"),
                          ("_loc_evidence", "loc_evidence"), ("_loc_mlp", "loc_mlp")):
            module = getattr(model, name, None)
            if module is not None and key in heads:
                module.load_state_dict(heads[key])
        if getattr(model, "_frame_null", None) is not None and "frame_null" in heads:
            model._frame_null.data = heads["frame_null"].to(device)
        return model, classes
