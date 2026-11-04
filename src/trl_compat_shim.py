"""trl compatibility shim for official 2024-era SPIN code. Project file (NOT official SPIN).

Official uclaml/SPIN (commit a12ba80) was written against trl~=0.7 and does::

    from trl.import_utils import is_peft_available, is_wandb_available
    from trl.models import PreTrainedModelWrapper, create_reference_model
    from trl.trainer.utils import disable_dropout_in_model, pad_to_length

The image installs modern trl (>=1.0, needed for Qwen3 SFT) where some of
these names were removed or relocated. This module re-attaches the missing
names -- reusing the modern equivalents when they exist, local fallbacks
otherwise -- so official ``run_spin.py`` imports cleanly.

Rules:
- Import-only compatibility. Never touches SPIN loss / trainer logic.
- Must be imported BEFORE any ``alignment.*`` module (see
  ``app.run_spin_official``, which pre-imports it in a launcher process).
- Idempotent: never overwrites a name that already exists.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types


def _pkg_available(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except Exception:
        return False


def _patch(target_mod: str, target_attr: str, candidates, fallback_factory) -> None:
    """Ensure ``target_mod.target_attr`` exists.

    ``candidates``: list of ``(modname, attr)`` tried first (modern
    relocations of the same helper). If none import, use ``fallback_factory``.
    """
    try:
        mod = importlib.import_module(target_mod)
    except Exception:
        mod = types.ModuleType(target_mod)
        sys.modules[target_mod] = mod
    if getattr(mod, target_attr, None) is not None:
        return
    for modname, attr in candidates:
        try:
            impl = getattr(importlib.import_module(modname), attr)
            setattr(mod, target_attr, impl)
            print(f"[trl-compat] {target_mod}.{target_attr} <- {modname}.{attr}", flush=True)
            return
        except Exception:
            continue
    setattr(mod, target_attr, fallback_factory())
    print(f"[trl-compat] {target_mod}.{target_attr} <- local fallback", flush=True)


def _wrapper_fallback():
    class PreTrainedModelWrapper:
        """Placeholder. Modern trl removed this class; official SPIN only
        uses it in type hints (never instantiated on the run_spin path)."""

    PreTrainedModelWrapper.__name__ = "PreTrainedModelWrapper"
    PreTrainedModelWrapper.__qualname__ = "PreTrainedModelWrapper"
    return PreTrainedModelWrapper


def _pad_fallback():
    import torch

    def pad_to_length(tensor, length, pad_value=0, dim=-1):
        """Same semantics as trl~=0.7's pad_to_length (used by SPINTrainer)."""
        if tensor.size(dim) >= length:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)],
            dim=dim,
        )

    return pad_to_length


_patch("trl.import_utils", "is_peft_available",
       [("trl.trainer.utils", "is_peft_available")],
       lambda: (lambda: _pkg_available("peft")))
_patch("trl.import_utils", "is_wandb_available", [],
       lambda: (lambda: _pkg_available("wandb")))
_patch("trl.models", "PreTrainedModelWrapper", [], _wrapper_fallback)
_patch("trl.trainer.utils", "pad_to_length", [], _pad_fallback)


def _transformers5_compat() -> None:
    """transformers 5.x removed the `use_flash_attention_2` from_pretrained
    kwarg, but official run_spin.py (2024) still passes it. Drop it here
    (mapping True -> attn_implementation) instead of editing official code."""
    try:
        from transformers import AutoModelForCausalLM
    except Exception:
        return
    orig = AutoModelForCausalLM.from_pretrained
    if getattr(orig, "_spin_compat_wrapped", False):
        return

    def from_pretrained(*args, **kwargs):
        ufa2 = kwargs.pop("use_flash_attention_2", None)
        if ufa2 and "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = "flash_attention_2"
        return orig(*args, **kwargs)

    from_pretrained._spin_compat_wrapped = True
    AutoModelForCausalLM.from_pretrained = from_pretrained
    print("[trl-compat] wrapped AutoModelForCausalLM.from_pretrained"
          " (drop use_flash_attention_2)", flush=True)


_transformers5_compat()


def _transformers5_trainer_compat() -> None:
    """transformers 5.x renamed the Trainer kwarg `tokenizer` to
    `processing_class` (and dropped the `Trainer.tokenizer` attribute).
    Official SPINTrainer (2024) still passes `tokenizer=` to super().__init__
    and reads `self.tokenizer`. Map both here; official code untouched."""
    try:
        from transformers import Trainer
    except Exception:
        return
    if not getattr(Trainer.__init__, "_spin_compat_wrapped", False):
        orig_init = Trainer.__init__

        def __init__(self, *args, **kwargs):
            if "tokenizer" in kwargs and "processing_class" not in kwargs:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
            return orig_init(self, *args, **kwargs)

        __init__._spin_compat_wrapped = True
        Trainer.__init__ = __init__
        print("[trl-compat] wrapped Trainer.__init__ (tokenizer -> processing_class)",
              flush=True)
    if not hasattr(Trainer, "tokenizer"):
        Trainer.tokenizer = property(lambda self: self.processing_class)
        print("[trl-compat] added Trainer.tokenizer property (-> processing_class)",
              flush=True)


_transformers5_trainer_compat()
