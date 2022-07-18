import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from argparse import Namespace
from typing import Sequence, Tuple, Dict

from jax_learning.buffers import ReplayBuffer
from jax_learning.buffers.utils import batch_flatten, to_jnp
from jax_learning.learners import Learner
from jax_learning.losses.policy_loss import sac_policy_loss
from jax_learning.losses.temperature_loss import sac_temperature_loss
from jax_learning.losses.value_loss import clipped_min_q_td_error
from jax_learning.models import StochasticPolicy, ActionValue, Temperature

import jax_learning.wandb_constants as w

Q_LOSS = "q_loss"
POLICY_LOSS = "policy_loss"
TEMPERATURE_LOSS = "temperature_loss"
MEAN_Q_LOSS = "mean_q_loss"
MEAN_POLICY_LOSS = "mean_policy_loss"
MEAN_TEMPERATURE_LOSS = "mean_temperature_loss"
MEAN_CURR_Q = "mean_curr_q"
MEAN_NEXT_Q = "mean_next_q"
MAX_CURR_Q = "max_curr_q"
MAX_NEXT_Q = "max_next_q"
MIN_CURR_Q = "min_curr_q"
MIN_NEXT_Q = "min_next_q"
MAX_TD_ERROR = "max_td_error"
MIN_TD_ERROR = "min_td_error"
POLICY = "policy"
V = "v"
TEMPERATURE = "temperature"
MEAN_TEMPERATURE = "mean_temperature"
TARGET_ENTROPY = "target_entropy"
OMEGA = "omega"


class PCL(Learner):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, opt, buffer, cfg)

        self._batch_size = cfg.batch_size
        self._num_gradient_steps = cfg.num_gradient_steps

        self._buffer_warmup = cfg.buffer_warmup
        self._actor_update_frequency = cfg.actor_update_frequency

        self._target_entropy = getattr(cfg, TARGET_ENTROPY, None)
        self._sample_key = jrandom.PRNGKey(cfg.seed)

        _clipped_min_q_td_error = jax.vmap(
            clipped_min_q_td_error, in_axes=[0, 0, 0, 0, 0, None, None]
        )

        @eqx.filter_grad(has_aux=True)
        def pc_loss(
            models: Tuple[StochasticPolicy, ActionValue],
            temperature: Temperature,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            dones: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
            keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            (policy, v) = models
            curr_xs = jnp.concatenate((obss, acts), axis=-1)
            curr_q_preds, _ = jax.vmap(q.q_values)(curr_xs, h_states)
            curr_q_preds_min = jnp.min(curr_q_preds, axis=1)

            next_acts, next_lprobs, _ = jax.vmap(policy.act_lprob)(
                next_obss, next_h_states, keys
            )
            next_lprobs = jnp.sum(next_lprobs, axis=-1, keepdims=True)

            next_xs = jnp.concatenate((next_obss, next_acts), axis=-1)
            next_q_preds, _ = jax.vmap(v.q_values)(next_xs, next_h_states)
            next_q_preds_min = jnp.min(next_q_preds, axis=1)

            temp = temperature()

            def batch_td_errors(curr_q_pred):
                return _clipped_min_q_td_error(
                    curr_q_pred,
                    next_q_preds_min,
                    next_lprobs,
                    rews,
                    dones,
                    temp,
                    self._gamma,
                )

            td_errors = jax.vmap(batch_td_errors, in_axes=[1])(curr_q_preds)
            loss = jnp.sum(jnp.mean(td_errors**2, axis=0))
            return loss, {
                Q_LOSS: loss,
                MAX_NEXT_Q: jnp.max(next_q_preds_min),
                MIN_NEXT_Q: jnp.min(next_q_preds_min),
                MEAN_NEXT_Q: jnp.mean(next_q_preds_min),
                MAX_CURR_Q: jnp.max(curr_q_preds_min),
                MIN_CURR_Q: jnp.min(curr_q_preds_min),
                MEAN_CURR_Q: jnp.mean(curr_q_preds_min),
                MAX_TD_ERROR: jnp.max(td_errors),
                MIN_TD_ERROR: jnp.min(td_errors),
                "max_q_log_prob": jnp.max(next_lprobs),
                "min_q_log_prob": jnp.min(next_lprobs),
                "mean_q_log_prob": jnp.mean(next_lprobs),
            }

        def update_models(
            models: Tuple[StochasticPolicy, ActionValue],
            temperature: Temperature,
            opt: Tuple[optax.GradientTransformation, optax.GradientTransformation],
            opt_state: Tuple[optax.OptState, optax.OptState],
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            dones: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
        ) -> Tuple[
            ActionValue,
            optax.OptState,
            Tuple[
                jax.tree_util.PyTreeDef,
                jax.tree_util.PyTreeDef,
                jax.tree_util.PyTreeDef,
            ],
            dict,
            jrandom.PRNGKey,
        ]:
            sample_key = jrandom.split(self._sample_key, num=1)[0]
            keys = jrandom.split(self._sample_key, num=self._batch_size)
            grads, learn_info = pc_loss(
                models,
                temperature,
                obss,
                h_states,
                acts,
                rews,
                dones,
                next_obss,
                next_h_states,
                keys,
            )
            (policy_grads, v_grads) = grads

            updates, opt_state = opt.update(grads, opt_state)
            q = eqx.apply_updates(q, updates)
            return (
                q,
                opt_state,
                grads,
                learn_info,
                sample_key,
            )

        @eqx.filter_grad(has_aux=True)
        def temperature_loss(
            temperature: Temperature,
            policy: StochasticPolicy,
            obss: np.ndarray,
            h_states: np.ndarray,
            keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            temp = temperature()
            _, lprobs, _ = jax.vmap(policy.act_lprob)(obss, h_states, keys)
            lprobs = jnp.sum(lprobs, axis=-1, keepdims=True)
            loss = jnp.mean(_sac_temperature_loss(temp, lprobs, self._target_entropy))
            return loss, {
                TEMPERATURE_LOSS: loss,
                TEMPERATURE: temp,
                "max_temperature_log_prob": jnp.max(lprobs),
                "min_temperature_log_prob": jnp.min(lprobs),
                "mean_temperature_log_prob": jnp.mean(lprobs),
            }

        def update_temperature(
            policy: StochasticPolicy,
            temperature: Temperature,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
        ) -> Tuple[
            ActionValue, optax.OptState, jax.tree_util.PyTreeDef, dict, jrandom.PRNGKey
        ]:
            sample_key = jrandom.split(self._sample_key, num=1)[0]
            keys = jrandom.split(self._sample_key, num=self._batch_size)
            grads, learn_info = temperature_loss(
                temperature, policy, obss, h_states, keys
            )

            updates, opt_state = opt.update(grads, opt_state)
            temperature = eqx.apply_updates(temperature, updates)
            return temperature, opt_state, grads, learn_info, sample_key

        self.update_models = eqx.filter_jit(update_models)
        self.update_temperature = eqx.filter_jit(update_temperature)

    def learn(self, next_obs: np.ndarray, next_h_state: np.ndarray, learn_info: dict):
        self._step += 1

        if (
            self._step <= self._buffer_warmup
            or (self._step - 1 - self._buffer_warmup) % self._update_frequency != 0
        ):
            return

        learn_info[f"{w.LOSSES}/{MEAN_Q_LOSS}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MEAN_CURR_Q}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MEAN_NEXT_Q}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MAX_CURR_Q}"] = -np.inf
        learn_info[f"{w.Q_VALUES}/{MAX_NEXT_Q}"] = -np.inf
        learn_info[f"{w.Q_VALUES}/{MIN_CURR_Q}"] = np.inf
        learn_info[f"{w.Q_VALUES}/{MIN_NEXT_Q}"] = np.inf
        learn_info[f"{w.ACTION_LOG_PROBS}/max_q_log_prob"] = 0.0
        learn_info[f"{w.ACTION_LOG_PROBS}/min_q_log_prob"] = 0.0
        learn_info[f"{w.ACTION_LOG_PROBS}/mean_q_log_prob"] = 0.0

        for update_i in range(self._num_gradient_steps):
            (
                obss,
                h_states,
                acts,
                rews,
                dones,
                next_obss,
                next_h_states,
                _,
                _,
                _,
            ) = self.buffer.sample_with_next_obs(
                batch_size=self._batch_size,
                next_obs=next_obs,
                next_h_state=next_h_state,
            )

            if self.obs_rms:
                obss = self.obs_rms.normalize(obss)

            (obss, h_states, acts, rews, dones, next_obss, next_h_states) = to_jnp(
                *batch_flatten(
                    obss, h_states, acts, rews, dones, next_obss, next_h_states
                )
            )
            (
                models,
                opt_state,
                grads,
                q_learn_info,
                self._sample_key,
            ) = self.update_models(
                models=(self.model[POLICY], self.model[V]),
                temperature=self.model[TEMPERATURE],
                opt=(self.opt[POLICY], self.opt[V]),
                opt_state=(self.opt_state[POLICY], self.opt_state[V]),
                obss=obss,
                h_states=h_states,
                acts=acts,
                rews=rews,
                dones=dones,
                next_obss=next_obss,
                next_h_states=next_h_states,
            )

            self._model[POLICY], self._model[V] = models
            self._opt_state[POLICY], self._opt_state[V] = opt_state

            if self._target_entropy is not None:
                learn_info(f"{w.LOSSES}/{MEAN_TEMPERATURE_LOSS}", 0.0)
                learn_info(f"{w.TRAIN}/{MEAN_TEMPERATURE}", 0.0)
                learn_info(f"{w.ACTION_LOG_PROBS}/max_temperature_log_prob", 0.0)
                learn_info(f"{w.ACTION_LOG_PROBS}/min_temperature_log_prob", 0.0)
                learn_info(f"{w.ACTION_LOG_PROBS}/mean_temperature_log_prob", 0.0)
                (
                    temperature,
                    opt_state,
                    grads,
                    temperature_learn_info,
                    self._sample_key,
                ) = self.update_temperature(
                    policy=self.model[POLICY],
                    temperature=self.model[TEMPERATURE],
                    opt=self.opt[TEMPERATURE],
                    opt_state=self.opt_state[TEMPERATURE],
                    obss=obss,
                    h_states=h_states,
                )
                self._model[TEMPERATURE] = temperature
                self._opt_state[TEMPERATURE] = opt_state

                learn_info[f"{w.TRAIN}/{MEAN_TEMPERATURE}"] += (
                    temperature_learn_info[TEMPERATURE].item()
                    / self._num_gradient_steps
                )
                learn_info[f"{w.LOSSES}/{MEAN_TEMPERATURE_LOSS}"] += (
                    temperature_learn_info[TEMPERATURE_LOSS].item()
                    / self._num_gradient_steps
                )
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/max_temperature_log_prob"
                ] = temperature_learn_info["max_temperature_log_prob"]
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/min_temperature_log_prob"
                ] = temperature_learn_info["min_temperature_log_prob"]
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/mean_temperature_log_prob"
                ] = temperature_learn_info["mean_temperature_log_prob"]

            learn_info[f"{w.LOSSES}/{MEAN_Q_LOSS}"] += (
                q_learn_info[Q_LOSS].item() / self._num_gradient_steps
            )

            learn_info[f"{w.Q_VALUES}/{MEAN_CURR_Q}"] += (
                q_learn_info[MEAN_CURR_Q].item() / self._num_gradient_steps
            )
            learn_info[f"{w.Q_VALUES}/{MEAN_NEXT_Q}"] += (
                q_learn_info[MEAN_NEXT_Q].item() / self._num_gradient_steps
            )
            learn_info[f"{w.Q_VALUES}/{MAX_CURR_Q}"] = max(
                learn_info[f"{w.Q_VALUES}/{MAX_CURR_Q}"],
                q_learn_info[MAX_CURR_Q].item(),
            )
            learn_info[f"{w.Q_VALUES}/{MAX_NEXT_Q}"] = max(
                learn_info[f"{w.Q_VALUES}/{MAX_NEXT_Q}"],
                q_learn_info[MAX_NEXT_Q].item(),
            )
            learn_info[f"{w.Q_VALUES}/{MIN_CURR_Q}"] = min(
                learn_info[f"{w.Q_VALUES}/{MIN_CURR_Q}"],
                q_learn_info[MIN_CURR_Q].item(),
            )
            learn_info[f"{w.Q_VALUES}/{MIN_NEXT_Q}"] = min(
                learn_info[f"{w.Q_VALUES}/{MIN_NEXT_Q}"],
                q_learn_info[MIN_NEXT_Q].item(),
            )

            learn_info[f"{w.ACTION_LOG_PROBS}/max_q_log_prob"] = q_learn_info[
                "max_q_log_prob"
            ]
            learn_info[f"{w.ACTION_LOG_PROBS}/min_q_log_prob"] = q_learn_info[
                "min_q_log_prob"
            ]
            learn_info[f"{w.ACTION_LOG_PROBS}/mean_q_log_prob"] = q_learn_info[
                "mean_q_log_prob"
            ]
