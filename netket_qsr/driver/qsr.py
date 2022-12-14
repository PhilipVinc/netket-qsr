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

from typing import List, Optional, Tuple, Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from numba import njit

from netket import jax as nkjax
from netket_qsr import jax as _nkjax
from netket.driver import AbstractVariationalDriver
from netket.driver.vmc_common import info
from netket.operator import AbstractOperator, LocalOperator
from netket.vqs import VariationalState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils import mpi
from netket.utils.types import PyTree

BaseType = Union[AbstractOperator, np.ndarray, str]


def _build_rotation(hi, basis, dtype=complex):
    localop = LocalOperator(hi, constant=1.0, dtype=dtype)
    U_X = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    U_Y = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, -1j], [1.0, 1j]])

    assert len(basis) == hi.size

    for (j, base) in enumerate(basis):
        if base == "X":
            localop *= LocalOperator(hi, U_X, [j])
        elif base == "Y":
            localop *= LocalOperator(hi, U_Y, [j])
        elif base == "Z" or base == "I":
            pass

    return localop


def _check_bases_type(Us: Union[List[BaseType], np.ndarray]) -> List[AbstractOperator]:
    """Checks if the given bases are valid for the Qsr driver.

    Args:
        Us (list or np.ndarray): A list of bases

    Raises:
        ValueError: When not given a list or np.ndarray
        TypeError: If the type of the operators is not a child of AbstractOperator
    """
    if not (isinstance(Us, list) or isinstance(Us, np.ndarray)):
        raise ValueError(
            "The bases should be a list or np.ndarray(dtype=object)" " of the bases."
        )

    if isinstance(Us[0], AbstractOperator):
        return Us

    if isinstance(Us[0], str):
        from netket.hilbert import Spin

        hilbert = Spin(0.5, N=len(Us[0]))
        N_samples = len(Us)

        _cache = {}
        _bases = np.empty(N_samples, dtype=object)

        for (i, basis) in enumerate(Us):
            if basis not in _cache:
                U = _build_rotation(hilbert, basis)
                _cache[basis] = U

            _bases[i] = _cache[basis]
        return _bases

    raise TypeError("Unknown type of measurement basis.")


def _convert_data(sigma_s, Us):
    """Converts samples and rotation operators to a more direct computational format.

    Args:
        sigma_s (np.ndarray): The states
        Us (np.ndarray or list): The list of rotations
    """
    Us = _check_bases_type(Us)
    # TODO: add Error message when user tries to convert less or more sigmas than Us
    N = sigma_s.shape[-1]
    sigma_s = sigma_s.reshape(-1, N)
    Nb = sigma_s.shape[0]

    # constant number of connected states per opoerator
    Nc = Us[0].hilbert.local_size
    sigma_p = np.zeros((0, N), dtype=sigma_s.dtype)
    mels = np.zeros((0,), dtype=Us[0].dtype)
    secs = np.zeros(Nb + 1, dtype=np.intp)
    MAX_LEN = 0

    last_i = 0
    for (i, (sigma, U)) in enumerate(zip(sigma_s, Us)):
        sigma_p_i, mels_i = U.get_conn(sigma)
        Nc = mels_i.size
        sigma_p.resize((last_i + Nc, N))
        mels.resize((last_i + Nc,))
        sigma_p[last_i:, :] = sigma_p_i
        mels[last_i:] = mels_i
        secs[i] = last_i
        last_i = last_i + Nc
        MAX_LEN = max(Nc, MAX_LEN)

    # last
    sigma_p.resize((last_i + MAX_LEN, N))
    mels.resize((last_i + MAX_LEN,))
    sigma_p[last_i + Nc :, :] = 0.0
    mels[last_i + Nc :] = 0.0
    secs[-1] = last_i  # + MAX_LEN

    return sigma_p, mels, secs, MAX_LEN


@njit
def _compose_sampled_data(
    sigma_p, mels, secs, MAX_LEN, sampled_indices, min_padding_factor=128
):
    N_samples = sampled_indices.size
    N = sigma_p.shape[-1]

    _sigma_p = np.zeros((N_samples * MAX_LEN, N), dtype=sigma_p.dtype)
    _mels = np.zeros((N_samples * MAX_LEN,), dtype=mels.dtype)
    _secs = np.zeros((N_samples + 1,), dtype=secs.dtype)
    _maxlen = 0

    last_i = 0
    for (n, i) in enumerate(sampled_indices):
        start_i, end_i = secs[i], secs[i + 1]
        len_i = end_i - start_i

        # print(f"{n}, {i}, {last_i}, {start_i}, {end_i}")
        _sigma_p[last_i : last_i + len_i, :] = sigma_p[start_i:end_i, :]
        _mels[last_i : last_i + len_i] = mels[start_i:end_i]

        last_i = last_i + len_i
        _secs[n + 1] = last_i

        _maxlen = max(_maxlen, len_i)

    padding_factor = max(
        MAX_LEN, min_padding_factor
    )  # minimum padding at 64 to avoid excessive recmpilation
    padded_size = padding_factor * int(np.ceil(last_i / padding_factor))

    _sigma_p = _sigma_p[:padded_size, :]
    _mels = _mels[:padded_size]

    return _sigma_p, _mels, _secs, _maxlen


@partial(jax.jit, static_argnums=0)
def _avg_O(afun, pars, model_state, sigma):
    sigma = sigma.reshape((-1, sigma.shape[-1]))
    _, vjp = nkjax.vjp(lambda W: afun({"params": W, **model_state}, sigma), pars)
    (O_avg,) = vjp(jnp.ones(sigma.shape[0]) / sigma.shape[0])
    return jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


####


def sum_sections(arr, secs):
    """
    Equivalent to
    for i in range(secs.size-1):
        out[i] = jnp.sum(arr[secs[i]:secs[i+1]])

    secs must be 1 longer than the number of sections desired.
    MAX_SIZE is the max length of a section
    """
    offsets = secs[:-1]
    sizes = secs[1:] - offsets

    N_rows = secs.size - 1
    # Construct the indices necessary to perform the segment_sum
    indices = jnp.repeat(jnp.arange(N_rows), sizes, total_repeat_length=arr.size)

    return jax.ops.segment_sum(
        arr, indices, num_segments=N_rows, indices_are_sorted=True
    )


@partial(jax.jit, static_argnums=(0,))
def local_value_rotated_kernel(log_psi, pars, sigma_p, mel, secs):
    log_psi_sigma_p = log_psi(pars, sigma_p)
    U_sigma_sigma_p_psi_sigma_p = mel * jnp.exp(log_psi_sigma_p)

    return jnp.log(sum_sections(U_sigma_sigma_p_psi_sigma_p, secs))


@partial(jax.jit, static_argnums=(0,))
def grad_local_value_rotated(log_psi, pars, model_state, sigma_p, mel, secs):
    log_val_rotated, vjp = nkjax.vjp(
        lambda W: local_value_rotated_kernel(
            log_psi, {"params": W, **model_state}, sigma_p, mel, secs
        ),
        pars,
    )
    log_val_rotated, _ = mpi.mpi_mean_jax(log_val_rotated)

    (O_avg,) = vjp(jnp.ones_like(log_val_rotated) / log_val_rotated.size)

    O_avg = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)

    return log_val_rotated, O_avg


@jax.jit
def compose_grads(grad_neg, grad_pos):
    return jax.tree_util.tree_map(
        lambda x, y: 2.0 * jnp.conj(x - y),
        grad_neg,
        grad_pos,
    )


# for nll
@partial(jax.jit, static_argnums=(0,))
def local_value_rotated_amplitude(log_psi, pars, sigma_p, mel, secs):
    log_psi_sigma_p = log_psi(pars, sigma_p)
    U_sigma_sigma_p_psi_sigma_p = mel * jnp.exp(log_psi_sigma_p)

    return jnp.log(jnp.abs(sum_sections(U_sigma_sigma_p_psi_sigma_p, secs)) ** 2)


####


class QSR(AbstractVariationalDriver):
    """
    Quantum State reconstruction driver.
    """

    def __init__(
        self,
        training_data: Tuple[List, List],
        training_batch_size: int,
        optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: PreconditionerT = identity_preconditioner,
        seed: Optional[int] = None,
    ):
        """Initializes the QSR driver class.

        Args:
            training_data: A tuple of two lists.
            training_batch_size: The trainign batch size.
            optimizer: The optimizer to use. You can use optax optimizers or
                choose from the predefined optimizers netket offers.
            variational_state: The Variational state to optimise
            preconditioner: The preconditioner to use.
                Defaults to identity_preconditioner.
            seed: The RNG seed. Defaults to None.

        Raises:
            TypeError: If the training data is not a 2 element tuple.
        """

        super().__init__(variational_state, optimizer)

        self.preconditioner = preconditioner

        self._dp = None  # type: PyTree
        self._S = None
        self._sr_info = None

        if not isinstance(training_data, tuple) or len(training_data) != 2:
            raise TypeError("not a tuple of length 2")

        self._rng = np.random.default_rng(
            np.asarray(nkjax.mpi_split(nkjax.PRNGKey(seed)))
        )

        sigma_p, mels, secs, MAX_LEN = _convert_data(*training_data)
        self._training_samples, self._training_rotations = training_data
        self._training_sigma_p = sigma_p
        self._training_mels = mels
        self._training_secs = secs
        self._training_max_len = MAX_LEN
        self._training_samples_n = len(secs) - 1

        self.training_batch_size = training_batch_size

    def _forward_and_backward(self):
        state = self.state

        state.reset()

        self._grad_neg = _avg_O(
            state._apply_fun,
            state.parameters,
            state.model_state,
            state.samples,
        )

        # sample training data for pos grad
        self._sampled_indices = np.sort(
            self._rng.integers(
                self._training_samples_n, size=(self.training_batch_size,)
            )
        )

        # compose data
        self._sigma_p, self._mels, self._secs, self._maxlen = _compose_sampled_data(
            self._training_sigma_p,
            self._training_mels,
            self._training_secs,
            self._training_max_len,
            self._sampled_indices,
        )

        _log_val_rot, self._grad_pos = grad_local_value_rotated(
            state._apply_fun,
            state.parameters,
            state.model_state,
            self._sigma_p,
            self._mels,
            self._secs,
        )

        self._loss_grad = compose_grads(self._grad_neg, self._grad_pos)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    def nll(self):
        """
        Compute the Negative-Log-Likelihood.

        .. warn::

            Exponentially expensive in the hilbert space size!

        """
        log_val_rot = local_value_rotated_amplitude(
            self.state._apply_fun,
            self.state.variables,
            self._sigma_p,
            self._mels,
            self._secs,
        )

        ce = jnp.mean(log_val_rot)
        # mpi

        # log norm calculation
        log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        return log_n - ce

    def _log_additional_data(self, log_dict, step):
        log_dict["loss_grad_norm"] = _nkjax.tree_norm(self._loss_grad)
        log_dict["dp_norm"] = _nkjax.tree_norm(self._dp)

    def __repr__(self):
        return (
            "QSR("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
                ("State       ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
