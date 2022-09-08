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
from jax_learning.learners import ReinforcementLearner
from jax_learning.losses.temperature_loss import sac_temperature_loss
from jax_learning.losses.value_loss import path_consistency_error
from jax_learning.models import StochasticPolicy, Value, Temperature

import jax_learning.wandb_constants as w

PC_LOSS = "pc_loss"
TEMPERATURE_LOSS = "temperature_loss"
MEAN_TEMPERATURE_LOSS = "mean_temperature_loss"
MEAN_PC_LOSS = "mean_pc_loss"
MEAN_CURR_Q = "mean_curr_q"
MAX_CURR_Q = "max_curr_q"
MIN_CURR_Q = "min_curr_q"
MAX_TD_ERROR = "max_td_error"
MIN_TD_ERROR = "min_td_error"
POLICY = "policy"
V = "v"
TEMPERATURE = "temperature"
MEAN_TEMPERATURE = "mean_temperature"
TARGET_ENTROPY = "target_entropy"
OMEGA = "omega"
HORIZON_LENGTH = "horizon_length"


class PCL(ReinforcementLearner):
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

        self._target_entropy = getattr(cfg, TARGET_ENTROPY, None)
        self._sample_key = jrandom.PRNGKey(cfg.seed)

        self._horizon_length = getattr(cfg, HORIZON_LENGTH, 1) + 1

        _path_consistency_error = jax.vmap(
            path_consistency_error, in_axes=[0, 0, 0, 0, None, None]
        )

        @eqx.filter_grad(has_aux=True)
        def pc_loss(
            models: Tuple[StochasticPolicy, Value],
            temperature: Temperature,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            lengths: np.ndarray,
            keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            (policy, v) = models
            v_preds, _ = jax.vmap(jax.vmap(v.values))(obss, h_states)
            lprobs, _ = jax.vmap(jax.vmap(policy.lprob))(obss, h_states, acts)
            lprobs = jnp.sum(lprobs, axis=-1)

            temp = temperature()

            pcl_errors = _path_consistency_error(
                lprobs, v_preds, rews[..., 0], lengths, temp, self._gamma
            )
            loss = jnp.sum(jnp.mean(pcl_errors**2, axis=0))
            return loss, {
                PC_LOSS: loss,
            }

        def update_models(
            policy: StochasticPolicy,
            v: Value,
            temperature: Temperature,
            policy_opt: optax.GradientTransformation,
            v_opt: optax.GradientTransformation,
            policy_opt_state: optax.OptState,
            v_opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            lengths: np.ndarray,
        ) -> Tuple[
            StochasticPolicy,
            Value,
            optax.OptState,
            optax.OptState,
            Tuple[
                jax.tree_util.PyTreeDef,
                jax.tree_util.PyTreeDef,
            ],
            dict,
            jrandom.PRNGKey,
        ]:
            sample_key = jrandom.split(self._sample_key, num=1)[0]
            keys = jrandom.split(self._sample_key, num=self._batch_size)
            grads, learn_info = pc_loss(
                (policy, v),
                temperature,
                obss,
                h_states,
                acts,
                rews,
                lengths,
                keys,
            )
            (policy_grads, v_grads) = grads

            policy_updates, policy_opt_state = policy_opt.update(
                policy_grads, policy_opt_state
            )
            policy = eqx.apply_updates(policy, policy_updates)

            v_updates, v_opt_state = v_opt.update(v_grads, v_opt_state)
            v = eqx.apply_updates(v, v_updates)
            return (
                policy,
                v,
                policy_opt_state,
                v_opt_state,
                grads,
                learn_info,
                sample_key,
            )

        _sac_temperature_loss = jax.vmap(sac_temperature_loss, in_axes=[None, 0, None])

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
            Temperature, optax.OptState, jax.tree_util.PyTreeDef, dict, jrandom.PRNGKey
        ]:
            sample_key = jrandom.split(self._sample_key, num=1)[0]
            keys = jrandom.split(self._sample_key, num=len(obss))
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

        learn_info[f"{w.LOSSES}/{MEAN_PC_LOSS}"] = 0.0

        for update_i in range(self._num_gradient_steps):
            (
                obss,
                h_states,
                acts,
                rews,
                _,
                terminateds,
                _,
                _,
                lengths,
                sample_idxes,
            ) = self.buffer.sample(
                batch_size=self._batch_size,
                horizon_length=self._horizon_length,
            )

            if self.obs_rms:
                obss = self.obs_rms.normalize(obss)

            (obss, h_states, acts, rews, terminateds, lengths) = to_jnp(
                *batch_flatten(obss, h_states, acts, rews, terminateds, lengths)
            )

            (
                policy,
                v,
                policy_opt_state,
                v_opt_state,
                grads,
                pc_learn_info,
                self._sample_key,
            ) = self.update_models(
                policy=self.model[POLICY],
                v=self.model[V],
                temperature=self.model[TEMPERATURE],
                policy_opt=self.opt[POLICY],
                v_opt=self.opt[V],
                policy_opt_state=self.opt_state[POLICY],
                v_opt_state=self.opt_state[V],
                obss=obss.reshape(*sample_idxes.shape, -1),
                h_states=h_states.reshape(*sample_idxes.shape, -1),
                acts=acts.reshape(*sample_idxes.shape, -1),
                rews=rews.reshape(*sample_idxes.shape, -1),
                lengths=lengths,
            )

            self._model[POLICY], self._model[V] = policy, v
            self._opt_state[POLICY], self._opt_state[V] = policy_opt_state, v_opt_state

            if self._target_entropy is not None:
                learn_info.setdefault(f"{w.LOSSES}/{MEAN_TEMPERATURE_LOSS}", 0.0)
                learn_info.setdefault(f"{w.TRAIN}/{MEAN_TEMPERATURE}", 0.0)
                learn_info.setdefault(
                    f"{w.ACTION_LOG_PROBS}/max_temperature_log_prob", 0.0
                )
                learn_info.setdefault(
                    f"{w.ACTION_LOG_PROBS}/min_temperature_log_prob", 0.0
                )
                learn_info.setdefault(
                    f"{w.ACTION_LOG_PROBS}/mean_temperature_log_prob", 0.0
                )
                flattened_sample_mask = np.where(sample_idxes.flatten() != -1)
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
                    obss=obss[flattened_sample_mask],
                    h_states=h_states[flattened_sample_mask],
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

            learn_info[f"{w.LOSSES}/{MEAN_PC_LOSS}"] += (
                pc_learn_info[PC_LOSS].item() / self._num_gradient_steps
            )
