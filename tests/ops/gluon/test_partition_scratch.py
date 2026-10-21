"""No warp-specialised Gluon kernel uses compiler scratch memory inside its partitions.

Triton's shared-memory allocator places the scratch of an operation inside a partition (a layout conversion, a reduction
across warps) over the warp_specialize capture buffer, which the partitions read when they start; a partition that
starts after another one has written its scratch reads clobbered arguments (pointers, barrier and tensor-memory
addresses).  On an idle GPU the partitions start together and the window is not hit; when the SMs are shared (another
process on the same GPU) it is, and the kernel writes garbage (seen as non-finite states of the chunk-state kernel).

For each kernel the existing operator test that compiles it runs in a subprocess with an empty Triton cache and the MLIR
pass dump of that kernel; after shared-memory allocation, no operation inside the ``ttg.warp_specialize`` region may
carry an ``allocation.offset`` (every compiled variant is checked).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

HERE = Path(__file__).resolve().parent
KERNELS = [
    ("_chunk_states_kernel", "test_dendritic_chunk_states.py", "test_train_vs_reference or test_prefill_vs_reference"),
    ("_chunk_scan_fwd_kernel", "test_dendritic_chunk_scan_fwd.py", "test_vs_reference"),
    ("_chunk_scan_dx_kernel", "test_dendritic_chunk_scan_dx.py", "test_vs_reference"),
    ("_dda_kernel", "test_dendritic_dda.py", "test_dda_vs_reference"),
    ("_dstates_kernel", "test_dendritic_dstates.py", "test_dstates_vs_reference"),
]
AFTER_ALLOCATION = "IR Dump Before TritonTensorMemoryAllocationPass"


def _sections(dump: str):
    """The module IR after shared-memory allocation, once per compiled variant."""
    out, i = [], dump.find(AFTER_ALLOCATION)
    while i >= 0:
        j = dump.find("// -----// IR Dump", i + len(AFTER_ALLOCATION))
        out.append(dump[i:j if j > 0 else None])
        i = dump.find(AFTER_ALLOCATION, i + len(AFTER_ALLOCATION))
    return out


@pytest.mark.parametrize("kernel,test_file,select", KERNELS, ids=[k[0] for k in KERNELS])
def test_no_compiler_scratch_in_partitions(kernel, test_file, select, tmp_path):
    env = dict(os.environ, MLIR_ENABLE_DUMP=kernel, TRITON_CACHE_DIR=str(tmp_path / "triton"))
    # -s: the child pytest must not capture the compiler's stderr (the pass dump)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "-x", "-p", "no:cacheprovider", str(HERE / test_file),
                        "-k", select], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    sections = _sections(p.stdout + p.stderr)
    assert sections, f"{kernel}: no pass dump (the test did not compile it)"
    for ir in sections:
        lines = ir.splitlines()
        ws = [n for n, line in enumerate(lines) if "ttg.warp_specialize(" in line]
        assert len(ws) == 1, f"{kernel}: {len(ws)} warp_specialize ops"
        inside = [line.strip()[:160] for line in lines[ws[0] + 1:] if re.search(r"allocation\.offset = \d+", line)]
        assert not inside, f"{kernel}: compiler scratch inside the partitions: {inside[:4]}"
