import time
from dataclasses import dataclass
from typing import Tuple
import pdb
import jax
import jax.numpy as jnp
from jax import jit, lax
import equinox as eqx
import optax
from jax.random import PRNGKey
import distrax
import numpy as np
from functools import partial
from jax.scipy.stats import norm

@dataclass
class SimState:
    N: jnp.ndarray         # (num_envs, dim)
    t: int

def _simstate_flatten(s: SimState):
    children = (s.N)
    aux = None
    return children, aux

def _simstate_unflatten( children):
    return SimState(*children)

jax.tree_util.register_pytree_node(SimState, _simstate_flatten, _simstate_unflatten)


@dataclass
class SimParams:
    A: jnp.ndarray        # (dim, dim)
    K: jnp.ndarray        # scalar as jnp.array or float
    freqs: jnp.ndarray    # (num_envs, dim)
    dim: int

def _simparams_flatten(p: SimParams):
    children = (p.A, p.K, p.freqs, jnp.array(p.dim))
    aux = None
    return children, aux

def _simparams_unflatten(aux, children):
    A, K, freqs, dim_arr = children
    return SimParams(A=A, K=K, freqs=freqs, dim=dim_arr)

jax.tree_util.register_pytree_node(SimParams, _simparams_flatten, _simparams_unflatten)

# --------------------------
# Graph / Laplacian / Kcrit
# --------------------------

@jit
def laplacian_from_adjacency(adj: jnp.ndarray) -> jnp.ndarray:
    deg = jnp.sum(adj, axis=1)
    return jnp.diag(deg) - adj


@jit
def criticalK(adj: jnp.ndarray, data_size: int, scale: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    L = laplacian_from_adjacency(adj)
    eigvals = jnp.linalg.eigvalsh(L)
    eigvals = jnp.sort(eigvals)
    # small safety if graph disconnected -> second_smallest could be zero; clamp slightly to avoid divide-by-zero
    second_smallest = jnp.maximum(eigvals[1], 1e-12)
    largest = eigvals[-1]
    K = data_size * (jnp.pi ** 2 / 4.0) * jnp.sqrt(data_size) * largest / (scale * second_smallest ** 2)
    return K, L

# --------------------------
# Order parameter & reward
# --------------------------

@jit
def order_param(y: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
    """
    y: (num_envs, dim)
    L: (dim, dim)
    returns r: (num_envs,)
    """
 
    ysin = jnp.sin(y)  # shape (batch_size, N)
    ycos = jnp.cos(y)
   
    # Expand L to shape (batch_size, N, N)
    batch_size = y.shape[0]
    N = y.shape[1]
    L = jnp.broadcast_to(L, (batch_size, N, N))

    # Compute quadratic forms
    quad_form_sin = jnp.einsum('bi,bij,bj->b', ysin, L, ysin)
    quad_form_cos = jnp.einsum('bi,bij,bj->b', ycos, L, ycos)
  
  
    # Compute order parameter
    r = jnp.ones((batch_size)) - (quad_form_sin + quad_form_cos) / (N ** 2)
   
    return jnp.mean(r)

@jit
def count_reward_batch(state_N: jnp.ndarray, next_N: jnp.ndarray, action: jnp.ndarray, A: jnp.ndarray, R: float, dt: float):
    """
    state_N, next_N: (num_envs, dim)
    action: (num_envs, dim)
    A: (dim, dim)
    returns: (reward, r1, penalty) each (num_envs,)
    """
    def compute_r(y):
        # y: (num_envs, dim)
        phase_diff = y[:, :, None] - y[:, None, :]  # (B, d, d)
        force = (jnp.sin(phase_diff) ** 2) * A  # broadcast A
        return jnp.sum(force, axis=(1, 2)) / 2.0  # (B,)

    r0 = compute_r(state_N)
    r1 = compute_r(next_N)
    energy = jnp.sum(action ** 2, axis=1) * (dt)
    #jax.debug.print("🤯energy   {energy} 🤯", energy=jnp.mean(energy))
    penalty = energy * R
    reward = -(r1 - r0) - penalty
    return reward, r1, energy, penalty



@jit
def f_batch(y: jnp.ndarray, u: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    """
    y: (num_envs, dim)
    u: (num_envs, dim)
    params.freqs: (num_envs, dim)
    params.A: (dim, dim)
    """

    phase_diff = y[:, :, None] - y[:, None, :]  # (B, d, d)
    # params.A is (d,d) -> broadcast to (B,d,d)
    force_matrix = params.K * jnp.sin(phase_diff) * params.A
    force_sum = jnp.sum(force_matrix, axis=2)  # (B, d)
    force = u * force_sum / params.dim

    return force + params.freqs

@jit
def heun_step(y: jnp.ndarray, action: jnp.ndarray, params: SimParams, dt: float) -> jnp.ndarray:
    

    k1 = f_batch(y, action, params)
    k2 = f_batch(y + dt * k1 , action, params)
    return y + (dt / 2.0) * (k1 + k2 )
# --------------------------
# Step / simulate utilities
# --------------------------

@jit
def step_fn(state, action: jnp.ndarray, params: SimParams, L: jnp.ndarray, R: float, dt: float):
    """
    One step update for the whole batch.
    action: (num_envs, dim)
    returns: next_state, (reward, r1, penalty, order)
    """

    N_next = heun_step(state.N, action, params, dt)  # (num_envs, dim)
    next_state = SimState(N=N_next)
    reward, r1, penalty = count_reward_batch(state.N, N_next, action, params.A, R)
    order = order_param(N_next, L)
    return next_state, (reward, r1, penalty, order)


# --------------------------
# Init helpers
# --------------------------

def init_sim_params(key, A: jnp.ndarray, K: float, num_envs: int) -> SimParams:
    
    half_batch = num_envs // 2

    # 2. Define CDF bounds for [-1, 1]
    low_cdf = norm.cdf(-3.0, loc=0 )
    high_cdf = norm.cdf(3.0, loc=0)
    
    # 3. Sample Uniformly and transform via Inverse CDF (ppf)
    u = jax.random.uniform(key, shape=(half_batch, A.shape[0]))
    freq1 = norm.ppf(low_cdf + u * (high_cdf - low_cdf), loc=0)
    
    # 4. Create antithetic pairs
    freqs = jnp.concatenate([freq1, -freq1], axis=0)/5

    return SimParams(A=A.astype(jnp.float32), K=jnp.array(K, dtype=jnp.float32), freqs=freqs, dim=A.shape[0])

def activate_state(key, num_envs: int, dim: int) -> SimState:
    key, subkey = jax.random.split(key)

    half = num_envs // 2
    rand_half = jax.random.normal(subkey, shape=(half, dim), dtype=jnp.float32) / 5.0

    # Concatenate positive and negative pairs
    N0 = jnp.concatenate([rand_half, -rand_half], axis=0)

    
    #N0 = jax.random.normal(key, shape=(num_envs, dim), dtype=jnp.float32) / 5.0
    t0 = jnp.zeros((num_envs,), dtype=jnp.int32)
 
    return SimState(N=N0, t=t0)