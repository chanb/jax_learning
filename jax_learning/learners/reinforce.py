from argparse import Namespace
from typing import Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.learners import ReinforcementLearner
from jax_learning.losses.policy_loss import reinforce_loss
from jax_learning.losses.value_loss import monte_carlo_returns
from jax_learning.models import StochasticPolicy

import jax_learning.wandb_constants as w

POLICY = "policy"
LOSS = "loss"
MEAN_LOSS = "mean_loss"
MAX_RETURN = "max_return"
MIN_RETURN = "min_return"
MEAN_RETURN = "mean_return"
MAX_LOG_PROBS = "max_log_probs"
MIN_LOG_PROBS = "min_log_probs"
MEAN_LOG_PROBS = "mean_log_probs"


class REINFORCE(ReinforcementLearner):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, opt, buffer, cfg)

        self._sample_idxes = np.arange(self._update_frequency)

        @eqx.filter_grad(has_aux=True)
        def compute_loss(
            policy: StochasticPolicy,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rets: np.ndarray,
        ) -> Tuple[np.ndarray, dict]:
            lprobs, _ = jax.vmap(policy.lprob)(obss, h_states, acts)
            lprobs = jnp.sum(lprobs, axis=-1)
            loss = jnp.mean(jax.vmap(reinforce_loss)(lprobs, rets))
            return loss, {
                LOSS: loss,
                MAX_RETURN: jnp.max(rets),
                MIN_RETURN: jnp.min(rets),
                MEAN_RETURN: jnp.mean(rets),
                MAX_LOG_PROBS: jnp.max(lprobs),
                MIN_LOG_PROBS: jnp.min(lprobs),
                MEAN_LOG_PROBS: jnp.mean(lprobs),
            }

        def update_policy(
            policy: StochasticPolicy,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rets: np.ndarray,
        ) -> Tuple[StochasticPolicy, optax.OptState, jax.tree_util.PyTreeDef, dict]:
            grads, learn_info = compute_loss(policy, obss, h_states, acts, rets)

            updates, opt_state = opt.update(grads, opt_state)
            policy = eqx.apply_updates(policy, updates)
            return policy, opt_state, grads, learn_info

        self.update_policy = eqx.filter_jit(update_policy)

    def learn(self, next_obs: np.ndarray, next_h_state: np.ndarray, learn_info: dict):
        self._step += 1

        if self._step % self._update_frequency != 0:
            return

        obss, h_states, acts, rews, dones, _, _, _ = self.buffer.sample(
            batch_size=self._update_frequency, idxes=self._sample_idxes
        )
        if self.obs_rms:
            obss = self.obs_rms.normalize(obss)

        rets = monte_carlo_returns(rews, dones, self._gamma)
        if self.val_rms:
            self.val_rms.update(rets)
            rets = self.val_rms.normalize(rets)

        (obss, h_states, acts, rets) = to_jnp(
            *batch_flatten(obss, h_states, acts, rets)
        )
        policy, opt_state, grads, curr_learn_info = self.update_policy(
            policy=self.model[POLICY],
            opt=self.opt[POLICY],
            opt_state=self.opt_state[POLICY],
            obss=obss,
            h_states=h_states,
            acts=acts,
            rets=rets,
        )

        self._model[POLICY] = policy
        self._opt_state[POLICY] = opt_state

        learn_info[f"{w.LOSSES}/{MEAN_LOSS}"] = curr_learn_info[LOSS].item()
        learn_info[f"{w.Q_VALUES}/{MIN_RETURN}"] = curr_learn_info[MIN_RETURN].item()
        learn_info[f"{w.Q_VALUES}/{MAX_RETURN}"] = curr_learn_info[MAX_RETURN].item()
        learn_info[f"{w.Q_VALUES}/{MEAN_RETURN}"] = curr_learn_info[MEAN_RETURN].item()
        learn_info[f"{w.ACTION_LOG_PROBS}/{MIN_LOG_PROBS}"] = curr_learn_info[
            MIN_LOG_PROBS
        ].item()
        learn_info[f"{w.ACTION_LOG_PROBS}/{MAX_LOG_PROBS}"] = curr_learn_info[
            MAX_LOG_PROBS
        ].item()
        learn_info[f"{w.ACTION_LOG_PROBS}/{MEAN_LOG_PROBS}"] = curr_learn_info[
            MEAN_LOG_PROBS
        ].item()
        self.buffer.clear()
