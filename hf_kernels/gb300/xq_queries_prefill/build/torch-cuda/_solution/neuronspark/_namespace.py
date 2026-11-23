"""The ``torch.library`` namespace of this copy of the operators.

The namespace is the top-level name of the package this module is imported in: ``neuronspark`` when the directory is
used as it is, the unique identifier of the kernel build when it is loaded as a Hugging Face kernel (the loader
imports the package under ``metadata.json``'s ``id``), so that several copies and versions of the operators can be
loaded in one process without replacing each other's ``torch.library`` registrations.
"""
from __future__ import annotations

import torch

NAMESPACE = __name__.partition(".")[0]
OPS = getattr(torch.ops, NAMESPACE)
