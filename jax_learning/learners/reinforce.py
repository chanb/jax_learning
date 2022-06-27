from argparse import Namespace
from typing import Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.distributions.utils import get_lprob
from jax_learning.learners import ReinforcementLearner
from jax_learning.losses.policy_loss import reinforce_score_function
from jax_learning.losses.value_loss import monte_carlo_returns

POLICY = "policy"
LOSS = "loss"
MEAN_LOSS = "mean_loss"
MAX_RETURN = "max_return"
MIN_RETURN = "min_return"


class REINFORCE(ReinforcementLearner):
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 opt: Dict[str, optax.GradientTransformation],
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        super().__init__(model, opt, buffer, cfg)
        
        self._sample_idxes = np.arange(self._update_frequency)
        
        @eqx.filter_grad(has_aux=True)
        def reinforce_loss(policy: eqx.Module,
                           obss: np.ndarray,
                           h_states: np.ndarray,
                           acts: np.ndarray,
                           rets: np.ndarray) -> Tuple[np.ndarray, dict]:
            dists = jax.vmap(policy.dist)(obss, h_states)
            lprobs = jax.vmap(get_lprob)(dists, acts)
            scores = jax.vmap(reinforce_score_function)(lprobs, rets)
            loss = -jnp.mean(scores)
            return loss, {
                LOSS: loss,
                MAX_RETURN: jnp.max(rets),
                MIN_RETURN: jnp.min(rets),
            }
        
        def step(policy: eqx.Module,
                 opt: optax.GradientTransformation,
                 opt_state: optax.OptState,
                 obss: np.ndarray,
                 h_states: np.ndarray,
                 acts: np.ndarray,
                 rets: np.ndarray) -> Tuple[eqx.Module, optax.OptState, jax.tree_util.PyTreeDef, dict]:
            grads, learn_info = reinforce_loss(policy,
                                               obss,
                                               h_states,
                                               acts,
                                               rets)

            updates, opt_state = opt.update(grads, opt_state)
            policy = eqx.apply_updates(policy, updates)
            return policy, opt_state, grads, learn_info
        self.step = eqx.filter_jit(step)

    def learn(self,
              next_obs: np.ndarray,
              next_h_state: np.ndarray,
              learn_info: dict):
        self._step += 1
        
        if self._step % self._update_frequency != 0:
            return

        obss, h_states, acts, rews, dones, _, _, _ = self.buffer.sample(batch_size=self._update_frequency,
                                                                        idxes=self._sample_idxes)
        if self.obs_rms:
            obss = self.obs_rms.normalize(obss)

        rets = monte_carlo_returns(rews, dones, self._gamma)
        if self.val_rms:
            self.val_rms.update(rets)
            rets = self.val_rms.normalize(rets)

        (obss, h_states, acts, rets) = to_jnp(*batch_flatten(obss,
                                                             h_states,
                                                             acts,
                                                             rets))
        policy, opt_state, grads, curr_learn_info = self.step(policy=self.model[POLICY],
                                                              opt=self.opt[POLICY],
                                                              opt_state=self.opt_state[POLICY],
                                                              obss=obss,
                                                              h_states=h_states,
                                                              acts=acts,
                                                              rets=rets)

        self._model[POLICY] = policy
        self._opt_state[POLICY] = opt_state

        learn_info[MEAN_LOSS] = curr_learn_info[LOSS].item()
        self.buffer.clear()
