# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
from typing import Optional, Tuple, Callable

import numpy as np

from jax import numpy as jnp

from netket.utils.types import PyTree, Scalar

from netket.jax import tree_dot, tree_conj

@jax.jit
def tree_norm(a: PyTree, *, p: int = 2) -> Scalar:
    r"""
    compute the L-p vector norm of a PyTree, interpreted as a vector.

    Args:
        a: A PyTree vector
        p: the order of the norm to compute. Defaults to 2 (the L2 or Froebnius norm).

    Returns:
        A scalar.
    """
    if p != 2:
        raise NotImplementedError("Tree_norm for p!=2 not yet implemented.")

    return jnp.sqrt(tree_dot(tree_conj(a), a))
