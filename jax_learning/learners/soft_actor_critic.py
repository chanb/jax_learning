import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax

from argparse import Namespace
from typing import Sequence, Tuple, Dict

from jax_learning.buffers.ram_buffers import TransitionNumPyBuffer
from jax_learning.buffers.utils import batch_flatten, to_jnp
from jax_learning.common import EpochSummary, polyak_average_generator
from jax_learning.learners import ReinforcementLearnerWithTargetNetwork
from jax_learning.losses.policy_loss import sac_policy_loss
from jax_learning.losses.temperature_loss import sac_temperature_loss
from jax_learning.losses.value_loss import clipped_min_q_td_error
from jax_learning.models import StochasticPolicy, ActionValue, Temperature

import jax_learning.wandb_constants as w

Q_LOSS = "q_loss"
CQL_REGULARIZATION = "cql_regularization"
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
Q = "q"
TEMPERATURE = "temperature"
MEAN_TEMPERATURE = "mean_temperature"
TARGET_ENTROPY = "target_entropy"
OMEGA = "omega"


class SAC(ReinforcementLearnerWithTargetNetwork):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        target_model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: TransitionNumPyBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, target_model, opt, buffer, cfg)

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
        def q_loss(
            models: Tuple[ActionValue, ActionValue],
            policy: StochasticPolicy,
            temperature: Temperature,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
            keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            (q, target_q) = models
            curr_xs = jnp.concatenate((obss, acts), axis=-1)
            curr_q_preds, _ = jax.vmap(q.q_values)(curr_xs, h_states)
            curr_q_preds_min = jnp.min(curr_q_preds, axis=1)

            next_acts, next_lprobs, _ = jax.vmap(policy.act_lprob)(
                next_obss, next_h_states, keys
            )
            next_lprobs = jnp.sum(next_lprobs, axis=-1, keepdims=True)

            next_xs = jnp.concatenate((next_obss, next_acts), axis=-1)
            next_q_preds, _ = jax.vmap(target_q.q_values)(next_xs, next_h_states)
            next_q_preds_min = jnp.min(next_q_preds, axis=1)

            temp = temperature()

            def batch_td_errors(curr_q_pred):
                return _clipped_min_q_td_error(
                    curr_q_pred,
                    next_q_preds_min,
                    next_lprobs,
                    rews,
                    terminateds,
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

        apply_residual_gradient = polyak_average_generator(getattr(cfg, OMEGA, 1.0))

        def update_q(
            q: ActionValue,
            target_q: ActionValue,
            policy: StochasticPolicy,
            temperature: Temperature,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
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
            grads, learn_info = q_loss(
                (q, target_q),
                policy,
                temperature,
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
                keys,
            )
            (q_grads, target_q_grads) = grads
            grads = jax.tree_map(apply_residual_gradient, q_grads, target_q_grads)

            updates, opt_state = opt.update(grads, opt_state)
            q = eqx.apply_updates(q, updates)
            return (
                q,
                opt_state,
                grads,
                learn_info,
                sample_key,
            )

        _sac_policy_loss = jax.vmap(sac_policy_loss, in_axes=[0, 0, None])

        @eqx.filter_grad(has_aux=True)
        def policy_loss(
            policy: StochasticPolicy,
            q: ActionValue,
            temperature: Temperature,
            obss: np.ndarray,
            h_states: np.ndarray,
            keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            acts, lprobs, _ = jax.vmap(policy.act_lprob)(obss, h_states, keys)
            lprobs = jnp.sum(lprobs, axis=-1, keepdims=True)
            curr_xs = jnp.concatenate((obss, acts), axis=-1)
            curr_q_preds, _ = jax.vmap(q.q_values)(curr_xs, h_states)
            curr_q_preds_min = jnp.min(curr_q_preds, axis=1)
            temp = temperature()

            loss = jnp.mean(_sac_policy_loss(curr_q_preds_min, lprobs, temp))
            return loss, {
                POLICY_LOSS: loss,
                "max_policy_log_prob": jnp.max(lprobs),
                "min_policy_log_prob": jnp.min(lprobs),
                "mean_policy_log_prob": jnp.mean(lprobs),
            }

        def update_policy(
            policy: StochasticPolicy,
            q: ActionValue,
            temperature: Temperature,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
        ) -> Tuple[
            StochasticPolicy,
            optax.OptState,
            jax.tree_util.PyTreeDef,
            dict,
            jrandom.PRNGKey,
        ]:
            sample_key = jrandom.split(self._sample_key, num=1)[0]
            keys = jrandom.split(self._sample_key, num=self._batch_size)

            grads, learn_info = policy_loss(
                policy, q, temperature, obss, h_states, keys
            )

            updates, opt_state = opt.update(grads, opt_state)
            policy = eqx.apply_updates(policy, updates)
            return policy, opt_state, grads, learn_info, sample_key

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
            keys = jrandom.split(self._sample_key, num=self._batch_size)
            grads, learn_info = temperature_loss(
                temperature, policy, obss, h_states, keys
            )

            updates, opt_state = opt.update(grads, opt_state)
            temperature = eqx.apply_updates(temperature, updates)
            return temperature, opt_state, grads, learn_info, sample_key

        self.update_q = eqx.filter_jit(update_q)
        self.update_policy = eqx.filter_jit(update_policy)
        self.update_temperature = eqx.filter_jit(update_temperature)

    def learn(
        self,
        learn_info: dict,
        epoch_summary: EpochSummary,
        **kwargs,
    ):
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
                _,
                terminateds,
                _,
                next_obss,
                next_h_states,
                _,
                _,
                _,
            ) = self.buffer.sample_with_next_obs(
                batch_size=self._batch_size,
            )

            if self.obs_rms:
                obss = self.obs_rms.normalize(obss)

            (
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
            ) = to_jnp(
                *batch_flatten(
                    obss,
                    h_states,
                    acts,
                    rews,
                    terminateds,
                    next_obss,
                    next_h_states,
                )
            )
            q, opt_state, grads, q_learn_info, self._sample_key = self.update_q(
                q=self.model[Q],
                target_q=self.target_model[Q],
                policy=self.model[POLICY],
                temperature=self.model[TEMPERATURE],
                opt=self.opt[Q],
                opt_state=self.opt_state[Q],
                obss=obss,
                h_states=h_states,
                acts=acts,
                rews=rews,
                terminateds=terminateds,
                next_obss=next_obss,
                next_h_states=next_h_states,
            )

            self._model[Q] = q
            self._opt_state[Q] = opt_state

            if self._step % self._actor_update_frequency == 0:
                learn_info.setdefault(f"{w.LOSSES}/{MEAN_POLICY_LOSS}", 0.0)
                learn_info.setdefault(f"{w.ACTION_LOG_PROBS}/max_policy_log_prob", 0.0)
                learn_info.setdefault(f"{w.ACTION_LOG_PROBS}/min_policy_log_prob", 0.0)
                learn_info.setdefault(f"{w.ACTION_LOG_PROBS}/mean_policy_log_prob", 0.0)
                (
                    policy,
                    opt_state,
                    grads,
                    policy_learn_info,
                    self._sample_key,
                ) = self.update_policy(
                    policy=self.model[POLICY],
                    q=self.model[Q],
                    temperature=self.model[TEMPERATURE],
                    opt=self.opt[POLICY],
                    opt_state=self.opt_state[POLICY],
                    obss=obss,
                    h_states=h_states,
                    acts=acts,
                )
                self._model[POLICY] = policy
                self._opt_state[POLICY] = opt_state

                learn_info[f"{w.LOSSES}/{MEAN_POLICY_LOSS}"] += (
                    policy_learn_info[POLICY_LOSS].item() / self._num_gradient_steps
                )
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/max_policy_log_prob"
                ] = policy_learn_info["max_policy_log_prob"]
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/min_policy_log_prob"
                ] = policy_learn_info["min_policy_log_prob"]
                learn_info[
                    f"{w.ACTION_LOG_PROBS}/mean_policy_log_prob"
                ] = policy_learn_info["mean_policy_log_prob"]

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

            if self._step % self._target_update_frequency == 0:
                self.update_target_model(model_key=Q)

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


class CQLSAC(SAC):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        target_model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: TransitionNumPyBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, target_model, opt, buffer, cfg)
        self._cql_alpha = cfg.alpha
        self._cql_num_action_samples = cfg.num_action_samples
        self._min_action = getattr(cfg, "min_action", -1.0)
        self._max_action = getattr(cfg, "max_action", 1.0)

        _clipped_min_q_td_error = jax.vmap(
            clipped_min_q_td_error, in_axes=[0, 0, 0, 0, 0, None, None]
        )

        def uniform_act_lprobs(key: jrandom.PRNGKey, num_samples: int):
            acts = jrandom.uniform(
                key,
                (num_samples, *cfg.act_dim),
                maxval=self._max_action - self._min_action,
            ) - ((self._max_action + self._min_action) / 2)
            lprobs = np.log(1 / (self._max_action - self._min_action))
            return acts, lprobs

        @eqx.filter_grad(has_aux=True)
        def q_loss(
            models: Tuple[ActionValue, ActionValue],
            policy: StochasticPolicy,
            temperature: Temperature,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
            cql_alpha: float,
            cql_num_action_samples: int,
            keys: Sequence[jrandom.PRNGKey],
            cql_keys: Sequence[jrandom.PRNGKey],
        ) -> Tuple[np.ndarray, dict]:
            (q, target_q) = models
            curr_xs = jnp.concatenate((obss, acts), axis=-1)
            curr_q_preds, _ = jax.vmap(q.q_values)(curr_xs, h_states)
            curr_q_preds_min = jnp.min(curr_q_preds, axis=1)

            next_acts, next_lprobs, _ = jax.vmap(policy.act_lprob)(
                next_obss, next_h_states, keys
            )
            next_lprobs = jnp.sum(next_lprobs, axis=-1, keepdims=True)

            next_xs = jnp.concatenate((next_obss, next_acts), axis=-1)
            next_q_preds, _ = jax.vmap(target_q.q_values)(next_xs, next_h_states)
            next_q_preds_min = jnp.min(next_q_preds, axis=1)

            temp = temperature()

            def batch_td_errors(curr_q_pred):
                return _clipped_min_q_td_error(
                    curr_q_pred,
                    next_q_preds_min,
                    next_lprobs,
                    rews,
                    terminateds,
                    temp,
                    self._gamma,
                )

            td_errors = jax.vmap(batch_td_errors, in_axes=[1])(curr_q_preds)
            loss = jnp.sum(jnp.mean(td_errors**2, axis=0))

            cql_rand_acts, cql_rand_lprobs = uniform_act_lprobs(
                cql_keys[0], cql_num_action_samples
            )
            cql_rand_lprobs = jnp.sum(cql_rand_lprobs, axis=-1, keepdims=True)

            cql_curr_acts, cql_curr_lprobs, _ = jax.vmap(policy.act_lprob)(
                np.tile(obss, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(obss), *obss.shape[1:])
                ),
                np.tile(h_states, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(h_states), *h_states.shape[1:])
                ),
                cql_keys,
            )
            cql_curr_lprobs = jnp.sum(cql_curr_lprobs, axis=-1, keepdims=True)

            cql_next_acts, cql_next_lprobs, _ = jax.vmap(policy.act_lprob)(
                np.tile(next_obss, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(next_obss), *obss.shape[1:])
                ),
                np.tile(next_h_states, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(next_h_states), *h_states.shape[1:])
                ),
                cql_keys,
            )
            cql_next_lprobs = jnp.sum(cql_next_lprobs, axis=-1, keepdims=True)

            cql_rand_xs = jnp.concatenate(
                (
                    np.tile(obss, cql_num_action_samples).reshape(
                        (cql_num_action_samples * len(obss), *obss.shape[1:])
                    ),
                    cql_rand_acts,
                ),
                axis=-1,
            )
            cql_rand_q_preds, _ = jax.vmap(q.q_values)(
                cql_rand_xs,
                np.tile(h_states, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(h_states), *h_states.shape[1:])
                ),
            )

            cql_curr_xs = jnp.concatenate(
                (
                    np.tile(obss, cql_num_action_samples).reshape(
                        (cql_num_action_samples * len(obss), *obss.shape[1:])
                    ),
                    cql_curr_acts,
                ),
                axis=-1,
            )
            cql_curr_q_preds, _ = jax.vmap(q.q_values)(
                cql_curr_xs,
                np.tile(h_states, cql_num_action_samples).reshape(
                    (cql_num_action_samples * len(h_states), *h_states.shape[1:])
                ),
            )

            cql_next_xs = jnp.concatenate(
                (
                    np.tile(next_obss, cql_num_action_samples).reshape(
                        (cql_num_action_samples * len(next_obss), *next_obss.shape[1:])
                    ),
                    cql_next_acts,
                ),
                axis=-1,
            )
            cql_next_q_preds, _ = jax.vmap(q.q_values)(
                cql_next_xs,
                np.tile(next_h_states, cql_num_action_samples).reshape(
                    (
                        cql_num_action_samples * len(next_h_states),
                        *next_h_states.shape[1:],
                    )
                ),
            )

            all_cql_q1 = jnp.concatenate(
                (
                    cql_rand_q_preds[:, 0] - cql_rand_lprobs,
                    cql_curr_q_preds[:, 0] - cql_curr_lprobs,
                    cql_next_q_preds[:, 0] - cql_next_lprobs,
                )
            )
            all_cql_q2 = jnp.concatenate(
                (
                    cql_rand_q_preds[:, 1] - cql_rand_lprobs,
                    cql_curr_q_preds[:, 1] - cql_curr_lprobs,
                    cql_next_q_preds[:, 1] - cql_next_lprobs,
                )
            )
            cql_reg = cql_alpha * (
                jnp.mean(jnp.logaddexp(all_cql_q1 / cql_alpha))
                + jnp.mean(jnp.logaddexp(all_cql_q2 / cql_alpha))
                - jnp.sum(jnp.mean(curr_q_preds, axis=0))
            )

            return loss + cql_reg, {
                Q_LOSS: loss,
                CQL_REGULARIZATION: cql_reg,
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

        apply_residual_gradient = polyak_average_generator(getattr(cfg, OMEGA, 1.0))

        def update_q(
            q: ActionValue,
            target_q: ActionValue,
            policy: StochasticPolicy,
            temperature: Temperature,
            opt: optax.GradientTransformation,
            opt_state: optax.OptState,
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
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
            cql_keys = jrandom.split(
                self._sample_key, num=self._batch_size * self.cql_num_action_samples
            )
            grads, learn_info = q_loss(
                (q, target_q),
                policy,
                temperature,
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
                self._cql_num_action_samples,
                self._cql_alpha,
                keys,
                cql_keys,
            )
            (q_grads, target_q_grads) = grads
            grads = jax.tree_map(apply_residual_gradient, q_grads, target_q_grads)

            updates, opt_state = opt.update(grads, opt_state)
            q = eqx.apply_updates(q, updates)
            return (
                q,
                opt_state,
                grads,
                learn_info,
                sample_key,
            )
