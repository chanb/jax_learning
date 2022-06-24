from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def one_step_bellman_optimality_error(q_curr_pred: np.ndarray,
                                      act: np.ndarray,
                                      q_next_pred: np.ndarray,
                                      rew: np.ndarray,
                                      done: np.ndarray,
                                      gamma: np.ndarray) -> np.ndarray:
    q_target = rew + (1 - done) * (gamma * jnp.max(q_next_pred))
    return q_curr_pred[act] - q_target

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
    
    td_errors = jax.vmap(one_step_bellman_optimality_error)(q_curr_preds, acts, q_next_preds, rews, dones, gammas)
    loss = jnp.mean(td_errors ** 2)
    return loss, {
        "loss": loss,
        "max_q_next": jnp.max(q_next_preds),
        "min_q_next": jnp.min(q_next_preds),
        "mean_q_next": jnp.mean(q_next_preds),
        "max_q_curr": jnp.max(q_curr_preds),
        "min_q_curr": jnp.min(q_curr_preds),
        "mean_q_curr": jnp.mean(q_curr_preds),
        "max_td_error": jnp.max(td_errors),
        "min_td_error": jnp.min(td_errors),
    }
