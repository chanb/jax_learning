from argparse import Namespace
from jax.scipy.special import logsumexp
from typing import Tuple, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_learning.buffers.ram_buffers import TransitionNumPyBuffer
from jax_learning.buffers.utils import to_jnp, batch_flatten
from jax_learning.common import EpochSummary, polyak_average_generator
from jax_learning.learners import ReinforcementLearnerWithTargetNetwork
from jax_learning.losses.value_loss import q_learning_td_error
from jax_learning.models import ActionValue

import jax_learning.wandb_constants as w

LOSS = "loss"
CQL_REGULARIZATION = "cql_regularization"
MEAN_LOSS = "mean_loss"
MEAN_CURR_Q = "mean_curr_q"
MEAN_NEXT_Q = "mean_next_q"
MAX_CURR_Q = "max_curr_q"
MAX_NEXT_Q = "max_next_q"
MIN_CURR_Q = "min_curr_q"
MIN_NEXT_Q = "min_next_q"
MAX_TD_ERROR = "max_td_error"
MIN_TD_ERROR = "min_td_error"
Q = "q"
OMEGA = "omega"


class QLearning(ReinforcementLearnerWithTargetNetwork):
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
        self._buffer_warmup = cfg.buffer_warmup
        self._num_gradient_steps = cfg.num_gradient_steps
        _q_learning_td_error = jax.vmap(
            q_learning_td_error, in_axes=[0, 0, 0, 0, 0, None]
        )

        @eqx.filter_grad(has_aux=True)
        def compute_loss(
            models: Tuple[ActionValue, ActionValue],
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
        ) -> Tuple[np.ndarray, dict]:
            (q, target_q) = models
            curr_q_preds, _ = jax.vmap(q.q_values)(obss, h_states)
            next_q_preds, _ = jax.vmap(target_q.q_values)(next_obss, next_h_states)

            td_errors = _q_learning_td_error(
                curr_q_preds, acts, next_q_preds, rews, terminateds, self._gamma
            )
            loss = jnp.mean(td_errors**2)
            return loss, {
                LOSS: loss,
                MAX_NEXT_Q: jnp.max(next_q_preds),
                MIN_NEXT_Q: jnp.min(next_q_preds),
                MEAN_NEXT_Q: jnp.mean(next_q_preds),
                MAX_CURR_Q: jnp.max(curr_q_preds),
                MIN_CURR_Q: jnp.min(curr_q_preds),
                MEAN_CURR_Q: jnp.mean(curr_q_preds),
                MAX_TD_ERROR: jnp.max(td_errors),
                MIN_TD_ERROR: jnp.min(td_errors),
            }

        apply_residual_gradient = polyak_average_generator(getattr(cfg, OMEGA, 1.0))

        def update_q(
            q: ActionValue,
            target_q: ActionValue,
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
        ]:
            grads, learn_info = compute_loss(
                (q, target_q),
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
            )

            (q_grads, target_q_grads) = grads
            grads = jax.tree_map(apply_residual_gradient, q_grads, target_q_grads)

            updates, opt_state = opt.update(grads, opt_state)
            q = eqx.apply_updates(q, updates)
            return q, opt_state, (grads, q_grads, target_q_grads), learn_info

        self.update_q = eqx.filter_jit(update_q)

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

        learn_info[f"{w.LOSSES}/{MEAN_LOSS}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MEAN_CURR_Q}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MEAN_NEXT_Q}"] = 0.0
        learn_info[f"{w.Q_VALUES}/{MAX_CURR_Q}"] = -np.inf
        learn_info[f"{w.Q_VALUES}/{MAX_NEXT_Q}"] = -np.inf
        learn_info[f"{w.Q_VALUES}/{MIN_CURR_Q}"] = np.inf
        learn_info[f"{w.Q_VALUES}/{MIN_NEXT_Q}"] = np.inf

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
            acts = acts.astype(np.int64)

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
            q, opt_state, grads, q_learn_info = self.update_q(
                q=self.model[Q],
                target_q=self.target_model[Q],
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

            if self._step % self._target_update_frequency == 0:
                self.update_target_model(model_key=Q)

            learn_info[f"{w.LOSSES}/{MEAN_LOSS}"] += (
                q_learn_info[LOSS].item() / self._num_gradient_steps
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


class CQL(QLearning):
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

        _q_learning_td_error = jax.vmap(
            q_learning_td_error, in_axes=[0, 0, 0, 0, 0, None]
        )

        @eqx.filter_grad(has_aux=True)
        def compute_loss(
            models: Tuple[ActionValue, ActionValue],
            obss: np.ndarray,
            h_states: np.ndarray,
            acts: np.ndarray,
            rews: np.ndarray,
            terminateds: np.ndarray,
            next_obss: np.ndarray,
            next_h_states: np.ndarray,
            cql_alpha: float,
        ) -> Tuple[np.ndarray, dict]:
            (q, target_q) = models
            curr_q_preds, _ = jax.vmap(q.q_values)(obss, h_states)
            next_q_preds, _ = jax.vmap(target_q.q_values)(next_obss, next_h_states)

            td_errors = _q_learning_td_error(
                curr_q_preds, acts, next_q_preds, rews, terminateds, self._gamma
            )
            loss = jnp.mean(td_errors**2)

            cql_reg = cql_alpha * jnp.mean(
                logsumexp(curr_q_preds, axis=1) - jnp.take(curr_q_preds, acts)
            )

            return loss + cql_reg, {
                LOSS: loss,
                CQL_REGULARIZATION: cql_reg,
                MAX_NEXT_Q: jnp.max(next_q_preds),
                MIN_NEXT_Q: jnp.min(next_q_preds),
                MEAN_NEXT_Q: jnp.mean(next_q_preds),
                MAX_CURR_Q: jnp.max(curr_q_preds),
                MIN_CURR_Q: jnp.min(curr_q_preds),
                MEAN_CURR_Q: jnp.mean(curr_q_preds),
                MAX_TD_ERROR: jnp.max(td_errors),
                MIN_TD_ERROR: jnp.min(td_errors),
            }

        apply_residual_gradient = polyak_average_generator(getattr(cfg, OMEGA, 1.0))

        def update_q(
            q: ActionValue,
            target_q: ActionValue,
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
        ]:
            grads, learn_info = compute_loss(
                (q, target_q),
                obss,
                h_states,
                acts,
                rews,
                terminateds,
                next_obss,
                next_h_states,
                self._cql_alpha,
            )

            (q_grads, target_q_grads) = grads
            grads = jax.tree_map(apply_residual_gradient, q_grads, target_q_grads)

            updates, opt_state = opt.update(grads, opt_state)
            q = eqx.apply_updates(q, updates)
            return q, opt_state, (grads, q_grads, target_q_grads), learn_info

        self.update_q = eqx.filter_jit(update_q)
