# Copyright (c) 2024, Tri Dao, Albert Gu.  (mamba_ssm; the source file itself carries no copyright line)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from mamba_ssm 2.3.2.post1, mamba_ssm/ops/triton/softplus.py
# (sha256 addcf96f306dd2c9bf1c9ef9feda42381a6e55ad906dd92e220598fcfec7b508), whole file, lines 1-15.
# Modified by NeuronSpark: this header only; the code below is the source file unchanged (no autotune, no imports
# to rewrite, nothing deleted).  See PROVENANCE.md.

import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:
    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))
