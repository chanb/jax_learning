import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from argparse import Namespace
from typing import Tuple, Dict

from jax_learning.buffers import TransitionNumPyBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.common import EpochSummary
from jax_learning.learners.learners import Learner
from jax_learning.losses.supervised_loss import squared_loss
from jax_learning.models import StochasticPolicy

import jax_learning.wandb_constants as w


POLICY = "policy"
LOSS = "loss"
MEAN_LOSS = "mean_loss"


class BC(Learner):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: TransitionNumPyBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, opt, buffer, cfg)

        self._batch_size = cfg.batch_size

        @eqx.filter_grad(has_aux=True)
        @eqx.filter_jit()
        def compute_loss(
            policy: StochasticPolicy,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts_e: np.ndarray,
        ) -> Tuple[np.ndarray, dict]:
            acts_pi, _ = jax.vmap(policy.deterministic_action)(obss, h_states)
            loss = jnp.mean(jnp.sum(jax.vmap(squared_loss)(acts_pi, acts_e), axis=-1))
            return loss, {
                LOSS: loss,
            }

        def update_policy(
            policy: StochasticPolicy,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts_e: np.ndarray,
        ) -> Tuple[StochasticPolicy, optax.OptState, jax.tree_util.PyTreeDef, dict]:
            grads, learn_info = compute_loss(policy, obss, h_states, acts_e)

            updates, opt_state = opt.update(grads, opt_state)
            policy = eqx.apply_updates(policy, updates)
            return policy, opt_state, grads, learn_info

        self.update_policy = eqx.filter_jit(update_policy)

    def learn(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        self._step += 1

        obss, h_states, acts_e, _, _, _, _, _, _, _ = self.buffer.sample(
            batch_size=self._batch_size,
        )
        if self.obs_rms:
            obss = self.obs_rms.normalize(obss)

        (obss, h_states, acts_e) = to_jnp(*batch_flatten(obss, h_states, acts_e))
        policy, opt_state, grads, curr_learn_info = self.update_policy(
            policy=self.model[POLICY],
            opt=self.opt[POLICY],
            opt_state=self.opt_state[POLICY],
            obss=obss,
            h_states=h_states,
            acts_e=acts_e,
        )

        self._model[POLICY] = policy
        self._opt_state[POLICY] = opt_state

        learn_info[f"{w.LOSSES}/{MEAN_LOSS}"] = curr_learn_info[LOSS].item()
