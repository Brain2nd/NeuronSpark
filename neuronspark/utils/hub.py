"""Where a checkpoint is: a repository on the Hugging Face Hub or a local path, as the files a load reads.

``NeuronSparkLMHeadModel.from_pretrained``, :func:`load_tokenizer` and :func:`export` take one of

* a **checkpoint directory**: the format of :mod:`neuronspark.utils.hf` (``config.json``, ``model.safetensors``, the
  tokenizer files);
* a **training line** of NeuronSpark-Trainer: one of its checkpoints (``.../checkpoints/u<update>/``:
  ``model.safetensors`` and ``trainer.json``, whose ``model_config`` holds the configuration fields the line set), its
  run directory (``checkpoints/latest`` names the newest checkpoint) or its backup (the run directory under ``run/``,
  the tokenizer under ``tokenizer/``);

each as a local path or, for a checkpoint directory and a line's backup, as a repository on the Hub (``"owner/name"``,
at ``revision`` when given).  A path that exists is read as a path; any other ``owner/name`` is looked up on the Hub.

From the Hub only the files of the load are fetched (a line's backup also holds its optimizer state and its data), once,
into the cache of ``huggingface_hub``, all from one commit; the weights are fetched for a model, not for a tokenizer.
A private repository is read with the key ``huggingface_hub`` already uses (its login, or ``HF_TOKEN``).

:func:`locate` returns the :class:`Source`: the configuration, the weights file, the tokenizer file.  A training
checkpoint's weights are the line's parameters (``model.safetensors``); a moving average the line keeps is part of its
optimizer state and is not read.  Nothing here imports torch.

:func:`export` writes the checkpoint directory of any source (the form to publish: the directory can be uploaded to the
Hub as it is) and is this module's command::

    python -m neuronspark.utils.hub SOURCE DIRECTORY [--revision REV] [--link]
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from neuronspark.models.config_neuronspark import MODEL_TYPE, NeuronSparkConfig

__all__ = [
    "CONFIG_NAME",
    "WEIGHTS_NAME",
    "TOKENIZER_NAME",
    "TOKENIZER_FILES",
    "PREFIX",
    "Source",
    "locate",
    "read_config_file",
    "read_weights_header",
    "check_weights_file",
    "load_tokenizer",
    "export",
    "main",
]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# the tokenizer file read by load_tokenizer (the ``tokenizers`` library's format)
TOKENIZER_NAME = "tokenizer.json"
# tokenizer files copied next to a saved checkpoint when present in the tokenizer directory
TOKENIZER_FILES = (
    TOKENIZER_NAME,
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)
# prefix of every tensor key of model.safetensors (the LM head model's backbone)
PREFIX = "backbone."

# a training line's names (NeuronSpark-Trainer: nstrain/ckpt.py, nstrain/backup.py)
_TRAINER = "trainer.json"
_CHECKPOINTS = "checkpoints"
_LATEST = "latest"
_BACKUP_RUN = "run"
_BACKUP_TOKENIZER = "tokenizer"

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# a safetensors header of a model is some hundred kilobytes; a length beyond this is not a header
_HEADER_MAX = 1 << 27

_PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class Source:
    """The files of one checkpoint.

    * ``kind``: ``"checkpoint"`` (a checkpoint directory) or ``"training"`` (a checkpoint of a training line);
    * ``origin``: what was asked for and where it led (for messages);
    * ``config``: the configuration (``config.json`` of a checkpoint directory; the fields a training line set, the
      other fields at their defaults);
    * ``weights``: ``model.safetensors`` (``None`` when :func:`locate` was told to leave the weights);
    * ``tokenizer``: the tokenizer file of the ``tokenizers`` library (``tokenizer.json`` of a checkpoint directory,
      the data line's tokenizer of a training line); ``None`` when the source has none;
    * ``tokenizer_files``: the tokenizer files to keep with the checkpoint (:data:`TOKENIZER_FILES` of a checkpoint
      directory; ``tokenizer`` alone for a training line).
    """

    kind: str
    origin: str
    config: NeuronSparkConfig
    weights: Optional[Path]
    tokenizer: Optional[Path]
    tokenizer_files: Tuple[Path, ...] = ()


# ------------------------------------------------------------------------------------------------ files
def read_config_file(path: _PathLike) -> NeuronSparkConfig:
    """The :class:`NeuronSparkConfig` of a configuration file: a JSON object with ``"model_type": "neuronspark"``,
    read by :meth:`NeuronSparkConfig.from_dict`."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: the configuration is not a JSON object")
    mt = d.get("model_type", None)
    if mt != MODEL_TYPE:
        raise ValueError(f"{path}: model_type {mt!r} is not {MODEL_TYPE!r}")
    return NeuronSparkConfig.from_dict(d)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: not a JSON object")
    return d


def read_weights_header(path: _PathLike) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """``(tensors, data_start, file_size)`` of a safetensors file, read from its header alone: ``tensors`` maps each
    key to its ``dtype``, ``shape`` and ``data_offsets`` (relative to ``data_start``)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        head = fh.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: not a safetensors file ({size} bytes)")
        (n,) = struct.unpack("<Q", head)
        if n > min(size - 8, _HEADER_MAX):
            raise ValueError(f"{path}: not a safetensors file (a header of {n} bytes in a file of {size})")
        try:
            body = json.loads(fh.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}: not a safetensors file (its header is not JSON: {e})") from None
    if not isinstance(body, dict):
        raise ValueError(f"{path}: not a safetensors file (its header is not a JSON object)")
    tensors = {k: v for k, v in body.items() if k != "__metadata__"}
    return tensors, 8 + int(n), int(size)


def check_weights_file(path: _PathLike, config: NeuronSparkConfig) -> Dict[str, int]:
    """Raise unless the safetensors file at ``path`` is whole and has the size of ``config``'s model: every key under
    :data:`PREFIX`, the configuration's tensor count and element count, the data the header describes all there.  Read
    from the header (no tensor is loaded; the load itself checks every key, shape, dtype and constant:
    :func:`neuronspark.utils.hf.verify_state_dict`).  Returns ``{"tensors", "elements", "bytes"}``."""
    tensors, start, size = read_weights_header(path)
    foreign = sorted(k for k in tensors if not k.startswith(PREFIX))
    if foreign:
        raise ValueError(f"{path}: {len(foreign)} keys outside {PREFIX!r}: {foreign[:5]}")
    elements = sum(math.prod(int(s) for s in t["shape"]) for t in tensors.values())
    want_t = int(config.n_state_tensors)
    want_e = int(config.n_params_expected) + int(config.n_buffers_expected)
    if len(tensors) != want_t or elements != want_e:
        raise ValueError(f"{path}: {len(tensors)} tensors of {elements} elements; the configuration's model has "
                         f"{want_t} of {want_e}")
    end = start + max((int(t["data_offsets"][1]) for t in tensors.values()), default=0)
    if size != end:
        raise ValueError(f"{path}: {size} bytes; its header describes {end} (an incomplete copy?)")
    return {"tensors": len(tensors), "elements": elements, "bytes": size}


# ------------------------------------------------------------------------------------------------ local paths
def _tokenizer_files(directory: Path) -> Tuple[Path, ...]:
    return tuple(directory / n for n in TOKENIZER_FILES if (directory / n).is_file())


def _pick_tokenizer(files: Sequence[str]) -> Optional[str]:
    """The tokenizer file among a line's ``tokenizer/`` entries: ``tokenizer.json``, else the only JSON file."""
    named = [f for f in files if f.rsplit("/", 1)[-1] == TOKENIZER_NAME]
    if named:
        return named[0]
    js = [f for f in files if f.endswith(".json")]
    return js[0] if len(js) == 1 else None


def _line_tokenizer(ck: Path, meta: Dict[str, Any]) -> Optional[Path]:
    """The tokenizer of a local training checkpoint: the backup's ``tokenizer/`` when the checkpoint is inside a
    backup, else the file the data line's ``state.json`` names (when this host has it)."""
    if ck.parent.name == _CHECKPOINTS and ck.parent.parent.name == _BACKUP_RUN:
        d = ck.parent.parent.parent / _BACKUP_TOKENIZER
        if d.is_dir():
            pick = _pick_tokenizer(sorted(p.name for p in d.iterdir() if p.is_file()))
            if pick is not None:
                return d / pick
    work = (meta.get("data") or {}).get("work")
    if isinstance(work, str) and (Path(work) / "state.json").is_file():
        named = _read_json(Path(work) / "state.json").get("tokenizer")
        if isinstance(named, str) and Path(named).is_file():
            return Path(named)
    return None


def _training(ck: Path, origin: str, meta: Dict[str, Any], weights: Optional[Path],
              tokenizer: Optional[Path]) -> Source:
    fields = meta.get("model_config")
    if not isinstance(fields, dict):
        raise ValueError(f"{origin}: {_TRAINER} of {ck.name} holds no model_config")
    return Source("training", origin, NeuronSparkConfig.from_dict(fields), weights, tokenizer,
                  () if tokenizer is None else (tokenizer,))


def _latest(checkpoints: Path) -> Path:
    name = (checkpoints / _LATEST).read_text(encoding="utf-8").strip()
    ck = checkpoints / name
    if not name or not (ck / _TRAINER).is_file():
        raise FileNotFoundError(f"{checkpoints / _LATEST} names {name!r}, which is not a checkpoint there")
    return ck


def _local(p: Path, origin: str) -> Source:
    if not p.is_dir():
        raise FileNotFoundError(f"{origin}: not a directory (a checkpoint directory, a training line's run directory, "
                                "one of its checkpoints or its backup is)")
    if (p / CONFIG_NAME).is_file():
        files = _tokenizer_files(p)
        tok = p / TOKENIZER_NAME
        return Source("checkpoint", origin, read_config_file(p / CONFIG_NAME), p / WEIGHTS_NAME,
                      tok if tok.is_file() else None, files)
    if (p / _TRAINER).is_file():
        ck = p
    elif (p / _CHECKPOINTS / _LATEST).is_file():
        ck = _latest(p / _CHECKPOINTS)
    elif (p / _BACKUP_RUN / _CHECKPOINTS / _LATEST).is_file():
        ck = _latest(p / _BACKUP_RUN / _CHECKPOINTS)
    else:
        raise FileNotFoundError(f"{origin}: neither a checkpoint directory ({CONFIG_NAME}) nor a training line "
                                f"({_TRAINER}, {_CHECKPOINTS}/{_LATEST} or {_BACKUP_RUN}/{_CHECKPOINTS}/{_LATEST})")
    meta = _read_json(ck / _TRAINER)
    where = origin if ck == p else f"{origin} (checkpoint {ck.name})"
    return _training(ck, where, meta, ck / WEIGHTS_NAME, _line_tokenizer(ck, meta))


# ------------------------------------------------------------------------------------------------ the Hub
def _hub(repo: str, revision: Optional[str], weights: bool) -> Source:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise ImportError(f"{repo!r} is not a local path; reading it from the Hugging Face Hub needs the "
                          "huggingface_hub package (pip install huggingface_hub)") from e
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:                                                  # huggingface_hub before 0.22
        from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = HfApi().repo_info(repo, revision=revision)
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(f"{repo!r}: neither a local path nor a repository on the Hugging Face Hub that this "
                                "account can read") from e
    sha = info.sha
    files = {s.rfilename for s in (info.siblings or ())}
    origin = f"{repo}@{str(sha)[:12]}"

    def get(name: str) -> Path:
        if name not in files:
            raise FileNotFoundError(f"{origin}: no {name}")
        return Path(hf_hub_download(repo, name, revision=sha))

    if CONFIG_NAME in files:
        config = read_config_file(get(CONFIG_NAME))
        toks = tuple(get(n) for n in TOKENIZER_FILES if n in files)
        tok = next((t for t in toks if t.name == TOKENIZER_NAME), None)
        return Source("checkpoint", origin, config, get(WEIGHTS_NAME) if weights else None, tok, toks)
    latest = f"{_BACKUP_RUN}/{_CHECKPOINTS}/{_LATEST}"
    if latest in files:
        name = get(latest).read_text(encoding="utf-8").strip()
        base = f"{_BACKUP_RUN}/{_CHECKPOINTS}/{name}"
        if not name or f"{base}/{_TRAINER}" not in files:
            raise FileNotFoundError(f"{origin}: {latest} names {name!r}, which is not a checkpoint there")
        meta = _read_json(get(f"{base}/{_TRAINER}"))
        pick = _pick_tokenizer(sorted(f for f in files if f.startswith(_BACKUP_TOKENIZER + "/") and f.count("/") == 1))
        return _training(Path(base), f"{origin} (checkpoint {name})", meta,
                         get(f"{base}/{WEIGHTS_NAME}") if weights else None, None if pick is None else get(pick))
    raise FileNotFoundError(f"{origin}: neither a checkpoint directory ({CONFIG_NAME}) nor a training line's backup "
                            f"({latest})")


def locate(name_or_path: _PathLike, revision: Optional[str] = None, weights: bool = True) -> Source:
    """The :class:`Source` of ``name_or_path`` (module docstring).  ``revision``: a branch, tag or commit of a Hub
    repository (default: its main branch); ``weights=False`` leaves the weights where they are (nothing of them is
    fetched)."""
    p = Path(name_or_path).expanduser()
    if p.exists():
        if revision is not None:
            raise ValueError(f"{name_or_path}: a local path has no revision ({revision!r})")
        src = _local(p, str(name_or_path))
        return src if weights else Source(src.kind, src.origin, src.config, None, src.tokenizer, src.tokenizer_files)
    name = os.fspath(name_or_path)
    if isinstance(name, str) and _REPO.match(name):
        return _hub(name, revision, weights)
    raise FileNotFoundError(f"{name_or_path}: no such path (and not an owner/name of the Hugging Face Hub)")


# ------------------------------------------------------------------------------------------------ tokenizer
def load_tokenizer(name_or_path: _PathLike, revision: Optional[str] = None):
    """The tokenizer of a checkpoint (``tokenizers.Tokenizer``), from any source :func:`locate` reads; the weights are
    not fetched.

    NeuronSpark-Trainer tokenizes a document without the tokenizer's special tokens and ends it with the end token, so
    a prompt for a model it trained is ``tokenizer.encode(text, add_special_tokens=False).ids``."""
    src = locate(name_or_path, revision=revision, weights=False)
    if src.tokenizer is None:
        raise FileNotFoundError(f"{src.origin}: no tokenizer file ({TOKENIZER_NAME})")
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise ImportError("load_tokenizer needs the tokenizers package (pip install tokenizers)") from e
    return Tokenizer.from_file(str(src.tokenizer))


# ------------------------------------------------------------------------------------------------ export
def _place(src: Path, dst: Path, link: bool) -> None:
    real = Path(os.path.realpath(src))
    if not link:
        shutil.copyfile(real, dst)
        return
    try:
        os.link(real, dst)
    except OSError as e:
        if e.errno in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise OSError(e.errno, f"cannot hard-link {real} to {dst} ({e.strerror}): a link needs both on one file "
                                   "system; export without link to copy") from None
        raise


def export(name_or_path: _PathLike, directory: _PathLike, revision: Optional[str] = None,
           link: bool = False) -> Dict[str, Any]:
    """Write the checkpoint directory of ``name_or_path`` (any source of :func:`locate`) into ``directory``:
    ``config.json`` with every field, ``model.safetensors`` (the source's file, byte for byte) and the tokenizer files
    (a training line's tokenizer as ``tokenizer.json``).

    The weights file is checked first (:func:`check_weights_file`); a ``directory`` that already holds one of the files
    is refused.  ``link``: hard-link the weights instead of copying them (the same file system; no second copy on
    disk).  Returns what was written: ``{"directory", "source", "kind", "files", "tensors", "elements", "bytes"}``."""
    src = locate(name_or_path, revision=revision)
    if src.weights is None or not src.weights.is_file():
        raise FileNotFoundError(f"{src.origin}: no {WEIGHTS_NAME}")
    summary = check_weights_file(src.weights, src.config)
    out = Path(directory).expanduser()
    if src.kind == "checkpoint":
        tokens = [(f, f.name) for f in src.tokenizer_files]
    else:
        tokens = [] if src.tokenizer is None else [(src.tokenizer, TOKENIZER_NAME)]
    names = [CONFIG_NAME, WEIGHTS_NAME] + [n for _, n in tokens]
    held = [n for n in names if (out / n).exists() or (out / n).is_symlink()]
    if held:
        raise FileExistsError(f"{out} already holds {held}")
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    try:                                                 # the weights first: the one step that can run out of room
        written.append(out / WEIGHTS_NAME)
        _place(src.weights, out / WEIGHTS_NAME, link)
        written.append(out / CONFIG_NAME)
        src.config.to_json(out / CONFIG_NAME)
        for f, n in tokens:
            written.append(out / n)
            shutil.copyfile(os.path.realpath(f), out / n)
    except BaseException:                                # leave nothing of a write that did not finish
        for f in written:
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        raise
    return dict(summary, directory=str(out), source=src.origin, kind=src.kind, files=names)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m neuronspark.utils.hub",
        description="Write a checkpoint directory (config.json, model.safetensors, the tokenizer): the form "
                    "NeuronSparkLMHeadModel.from_pretrained loads and the form to publish.")
    ap.add_argument("source", help="a checkpoint directory, a training line (its run directory, one of its "
                                   "checkpoints, its backup) or a repository on the Hugging Face Hub (owner/name)")
    ap.add_argument("directory", help="where to write (created; files it already holds are never replaced)")
    ap.add_argument("--revision", default=None, help="branch, tag or commit of a Hub repository")
    ap.add_argument("--link", action="store_true", help="hard-link the weights instead of copying them")
    a = ap.parse_args(argv)
    try:
        s = export(a.source, a.directory, revision=a.revision, link=a.link)
    except (OSError, ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{s['source']} ({s['kind']}) -> {s['directory']}")
    print(f"  {', '.join(s['files'])}")
    print(f"  {s['tensors']} tensors, {s['elements']:,} elements, {s['bytes']:,} bytes of weights")
    if not any(n == TOKENIZER_NAME for n in s["files"]):
        print(f"  no tokenizer: the source has none (put {TOKENIZER_NAME} next to the weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
