from argparse import Namespace

import equinox as eqx
import jax
import numpy as np
import optax

from jax_learning.buffers.buffers import ReplayBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.learners.learners import Learner
from jax_learning.losses.value_loss import q_learning_loss

MEAN_LOSS = "mean_loss"
MEAN_Q_CURR = "mean_q_curr"
MEAN_Q_NEXT = "mean_q_next"
MAX_Q_CURR = "max_q_curr"
MAX_Q_NEXT = "max_q_next"
MIN_Q_CURR = "min_q_curr"
MIN_Q_NEXT = "min_q_next"


class QLearning(Learner):
    def __init__(self,
                 model: eqx.Module,
                 target_model: eqx.Module,
                 opt: optax.GradientTransformation,
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        super().__init__(model, opt, buffer, cfg)
        self._target_model = target_model
        self._opt_state = self._opt.init(model)
        
        self._step = cfg.load_step
        
        self._batch_size = cfg.batch_size
        self._buffer_warmup = cfg.buffer_warmup
        self._num_gradient_steps = cfg.num_gradient_steps
        self._gamma = cfg.gamma
        self._tau = cfg.tau
        self._update_frequency = cfg.update_frequency
        self._target_update_frequency = cfg.target_update_frequency
        self._omega = cfg.omega
        
        def step(model,
                 target_model,
                 opt,
                 opt_state,
                 obss,
                 h_states,
                 acts,
                 rews,
                 dones,
                 next_obss,
                 next_h_states,
                 gammas,
                 omega):
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
    
    @property
    def target_model(self):
        return self._target_model
    
    @property
    def opt_state(self):
        return self._opt_state
        
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
            model, opt_state, grads, curr_learn_info = self.step(model=self.model,
                                                                 target_model=self.target_model,
                                                                 opt=self.opt,
                                                                 opt_state=self.opt_state,
                                                                 obss=obss,
                                                                 h_states=h_states,
                                                                 acts=acts,
                                                                 rews=rews,
                                                                 dones=dones,
                                                                 next_obss=next_obss,
                                                                 next_h_states=next_h_states,
                                                                 gammas=gammas,
                                                                 omega=self._omega)

            self._model = model
            self._opt_state = opt_state
            
            if self._step % self._target_update_frequency == 0:
                self._target_model = jax.tree_map(lambda p, tp: p * self._tau + tp * (1 - self._tau),
                                                  self.model,
                                                  self.target_model)
            
            learn_info[MEAN_LOSS] += curr_learn_info["loss"].item() / self._num_gradient_steps
            learn_info[MEAN_Q_CURR] += curr_learn_info[MEAN_Q_CURR].item() / self._num_gradient_steps
            learn_info[MEAN_Q_NEXT] += curr_learn_info[MEAN_Q_NEXT].item() / self._num_gradient_steps
            learn_info[MAX_Q_CURR] = max(learn_info[MAX_Q_CURR], curr_learn_info[MAX_Q_CURR].item())
            learn_info[MAX_Q_NEXT] = max(learn_info[MAX_Q_NEXT], curr_learn_info[MAX_Q_NEXT].item())
            learn_info[MIN_Q_CURR] = min(learn_info[MIN_Q_CURR], curr_learn_info[MIN_Q_CURR].item())
            learn_info[MIN_Q_NEXT] = min(learn_info[MIN_Q_NEXT], curr_learn_info[MIN_Q_NEXT].item())
