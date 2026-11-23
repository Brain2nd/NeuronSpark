# Copyright 2026 Co-developed by KDA (Kernel Design Agents)
from ._solution import kernel as _entrypoint

run = _entrypoint.run
__all__ = ["run"]
if hasattr(_entrypoint, "prepare"):
    prepare = _entrypoint.prepare
    __all__.append("prepare")
