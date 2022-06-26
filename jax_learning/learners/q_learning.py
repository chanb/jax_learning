from argparse import Namespace
from typing import Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.learners import LearnerWithTargetNetwork
from jax_learning.losses.value_loss import q_learning_td_error

LOSS = "loss"
MEAN_LOSS = "mean_loss"
MEAN_Q_CURR = "mean_q_curr"
MEAN_Q_NEXT = "mean_q_next"
MAX_Q_CURR = "max_q_curr"
MAX_Q_NEXT = "max_q_next"
MIN_Q_CURR = "min_q_curr"
MIN_Q_NEXT = "min_q_next"
MAX_TD_ERROR = "max_td_error"
MIN_TD_ERROR = "min_td_error"
Q = "q"


class QLearning(LearnerWithTargetNetwork):
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 target_model: Dict[str, eqx.Module],
                 opt: Dict[str, optax.GradientTransformation],
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        super().__init__(model, target_model, opt, buffer, cfg)
        
        self._step = cfg.load_step
        self._batch_size = cfg.batch_size
        self._buffer_warmup = cfg.buffer_warmup
        self._num_gradient_steps = cfg.num_gradient_steps
        self._gamma = cfg.gamma
        self._tau = cfg.tau
        self._update_frequency = cfg.update_frequency
        self._target_update_frequency = cfg.target_update_frequency
        self._omega = cfg.omega

        @eqx.filter_grad(has_aux=True)
        def q_learning_loss(models: Tuple[eqx.Module, eqx.Module],
                            obss: np.ndarray,
                            h_states: np.ndarray,
                            acts: np.ndarray,
                            rews: np.ndarray,
                            dones: np.ndarray,
                            next_obss: np.ndarray,
                            next_h_states: np.ndarray,
                            gammas: np.ndarray) -> Tuple[np.ndarray, dict]:
            (model, target_model) = models
            q_curr_preds, _ = jax.vmap(model.q_values)(obss, h_states)
            q_next_preds, _ = jax.vmap(target_model.q_values)(next_obss, next_h_states)
            
            td_errors = jax.vmap(q_learning_td_error)(q_curr_preds, acts, q_next_preds, rews, dones, gammas)
            loss = jnp.mean(td_errors ** 2)
            return loss, {
                LOSS: loss,
                MAX_Q_NEXT: jnp.max(q_next_preds),
                MIN_Q_NEXT: jnp.min(q_next_preds),
                MEAN_Q_NEXT: jnp.mean(q_next_preds),
                MAX_Q_CURR: jnp.max(q_curr_preds),
                MIN_Q_CURR: jnp.min(q_curr_preds),
                MEAN_Q_CURR: jnp.mean(q_curr_preds),
                MAX_TD_ERROR: jnp.max(td_errors),
                MIN_TD_ERROR: jnp.min(td_errors),
            }
        
        def step(model: eqx.Module,
                 target_model: eqx.Module,
                 opt: optax.GradientTransformation,
                 opt_state: optax.OptState,
                 obss: np.ndarray,
                 h_states: np.ndarray,
                 acts: np.ndarray,
                 rews: np.ndarray,
                 dones: np.ndarray,
                 next_obss: np.ndarray,
                 next_h_states: np.ndarray,
                 gammas: np.ndarray,
                 omega: float) -> Tuple[eqx.Module, optax.OptState, Tuple[jax.tree_util.PyTreeDef, jax.tree_util.PyTreeDef, jax.tree_util.PyTreeDef], dict]:
            grads, learn_info = q_learning_loss((model, target_model),
                                                obss,
                                                h_states,
                                                acts,
                                                rews,
                                                dones,
                                                next_obss,
                                                next_h_states,
                                                gammas)

            (model_grads, target_model_grads) = grads
            grads = jax.tree_map(lambda g, tg: g * omega + tg * (1 - omega),
                                 model_grads,
                                 target_model_grads)

            updates, opt_state = opt.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, (grads, model_grads, target_model_grads), learn_info
        self.step = eqx.filter_jit(step)
        
    def learn(self,
              next_obs: np.ndarray,
              next_h_state: np.ndarray,
              learn_info: dict):
        self._step += 1
        
        if self._step <= self._buffer_warmup or (self._step - 1 - self._buffer_warmup) % self._update_frequency != 0:
            return

        learn_info[MEAN_LOSS] = 0.
        learn_info[MEAN_Q_CURR] = 0.
        learn_info[MEAN_Q_NEXT] = 0.
        learn_info[MAX_Q_CURR] = -np.inf
        learn_info[MAX_Q_NEXT] = -np.inf
        learn_info[MIN_Q_CURR] = np.inf
        learn_info[MIN_Q_NEXT] = np.inf
        for update_i in range(self._num_gradient_steps):
            obss, h_states, acts, rews, dones, next_obss, next_h_states, _, _, _ = self.buffer.sample_with_next_obs(batch_size=self._batch_size,
                                                                                                                    next_obs=next_obs,
                                                                                                                    next_h_state=next_h_state)

            if self.obs_rms:
                obss = self.obs_rms.normalize(obss)
            acts = acts.astype(np.int64)
            gammas = np.ones(self._batch_size) * self._gamma
            
            (obss, h_states, acts, rews, dones, next_obss, next_h_states, gammas) = to_jnp(*batch_flatten(obss,
                                                                                                          h_states,
                                                                                                          acts,
                                                                                                          rews,
                                                                                                          dones,
                                                                                                          next_obss,
                                                                                                          next_h_states,
                                                                                                          gammas))
            model, opt_state, grads, curr_learn_info = self.step(model=self.model[Q],
                                                                 target_model=self.target_model[Q],
                                                                 opt=self.opt[Q],
                                                                 opt_state=self.opt_state[Q],
                                                                 obss=obss,
                                                                 h_states=h_states,
                                                                 acts=acts,
                                                                 rews=rews,
                                                                 dones=dones,
                                                                 next_obss=next_obss,
                                                                 next_h_states=next_h_states,
                                                                 gammas=gammas,
                                                                 omega=self._omega)

            self._model[Q] = model
            self._opt_state[Q] = opt_state
            
            if self._step % self._target_update_frequency == 0:
                self.polyak_average(model_key=Q)
            
            learn_info[MEAN_LOSS] += curr_learn_info[LOSS].item() / self._num_gradient_steps
            learn_info[MEAN_Q_CURR] += curr_learn_info[MEAN_Q_CURR].item() / self._num_gradient_steps
            learn_info[MEAN_Q_NEXT] += curr_learn_info[MEAN_Q_NEXT].item() / self._num_gradient_steps
            learn_info[MAX_Q_CURR] = max(learn_info[MAX_Q_CURR], curr_learn_info[MAX_Q_CURR].item())
            learn_info[MAX_Q_NEXT] = max(learn_info[MAX_Q_NEXT], curr_learn_info[MAX_Q_NEXT].item())
            learn_info[MIN_Q_CURR] = min(learn_info[MIN_Q_CURR], curr_learn_info[MIN_Q_CURR].item())
            learn_info[MIN_Q_NEXT] = min(learn_info[MIN_Q_NEXT], curr_learn_info[MIN_Q_NEXT].item())
