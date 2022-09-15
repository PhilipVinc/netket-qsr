# Copyright 2020 The Netket Authors. - All Rights Reserved.
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


import netket as nk
import netket_qsr as nkx

import numpy as np
import jax
import jax.numpy as jnp

from generate_data import generate

# Load the data
N = 3
hi, rotations, training_samples, ha, psi = generate(N, n_basis=2 * N, n_shots=500)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=complex)
# ma = nk.models.RBMModPhase(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1000, n_discard_per_chain=100)

## Quantum State Reconstruction
qst = nkx.driver.QSR(
    training_data=(training_samples, rotations),
    training_batch_size=500,
    optimizer=op,
    variational_state=vs,
    preconditioner=sr,
)

obs = {"Energy": ha}
E0 = psi.T.conj() @ (ha @ psi) / (psi.T.conj() @ psi)


def cb(step, logvals, driver):
    state = driver.state

    # Compute fidelity with exact state
    psima = state.to_array(normalize=True)
    fidelity = np.abs(np.vdot(psima, psi))

    # nll
    nll = driver.nll()

    msg = (
        f"\n{step:6} :   E = {logvals['Energy']} / {E0}\n"
        f"       :   F = {fidelity}\n"
        f"       : NLL = {nll}"
    )
    logvals["nll"] = nll
    logvals["fidelity"] = fidelity
    if nk.utils.mpi.rank == 0:
        print(msg, flush=True)
    return True


log = qst.run(
    n_iter=2000, obs=obs, callback=cb, step_size=5, out=nk.logging.RuntimeLog()
)
