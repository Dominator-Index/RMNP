"""
Activation recorder for the adaptive p-RMNP selector.

Records the input activation of every nn.Linear during forward passes that
run inside a `recorder.recording():` context. The recorded tensors feed the
row-curvature statistic b_i = ||u_i A||^2 in the p* criterion.

Interface (recording() context manager, get_activations()/clear(),
use_gram mode for distributed-friendly Gram storage) follows the design of
the ActivationRecorder in the SMuon reference implementation
(github.com/deel-ai-papers/schatten-muon, smuon/wrap_model.py); the code
here is an independent minimal reimplementation for this repo, which only
needs nn.Linear support (the GPT model has no conv layers).

Modes:
- use_gram=False (default): store the raw input activation tensor.
- use_gram=True: store the (in_features, in_features) Gram matrix A^T A.
  Grams from different data shards are additive, so they can be summed
  (all_reduce) across GPUs if global-batch curvature is ever wanted.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


class ActivationRecorder:
    """
    Registers forward hooks on all nn.Linear modules of `model` and, while
    recording is enabled, stores each layer's input keyed by its weight
    parameter (the same Parameter object the optimizer holds).

    Example
    -------
    >>> recorder = ActivationRecorder(model, use_gram=False)
    >>> with recorder.recording():
    ...     _ = model(x)
    >>> acts = recorder.get_activations()   # {weight Parameter -> Tensor}
    >>> recorder.clear()
    """

    def __init__(self, model: nn.Module, use_gram: bool = False):
        self.model = model
        self.use_gram = use_gram
        self.record_activations = False
        self._activations = {}
        self._hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        if not self.record_activations or not inputs:
            return
        weight = getattr(module, "weight", None)
        if weight is None:
            return
        act = inputs[0].detach()
        if self.use_gram:
            # (..., in_features) -> (samples, in_features); Gram = A^T A
            act2d = act.reshape(-1, act.size(-1)).float()
            self._activations[weight] = act2d.transpose(0, 1) @ act2d
        else:
            self._activations[weight] = act

    @contextmanager
    def recording(self):
        prev = self.record_activations
        self.record_activations = True
        try:
            yield self
        finally:
            self.record_activations = prev

    def get_activations(self):
        return self._activations

    def clear(self):
        self._activations.clear()

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self):
        try:
            self.remove_hooks()
            self.clear()
        except Exception:
            pass
