"""Baseline pricing policies. Each policy maps the env state to a multiplier per zone."""
import numpy as np


class Policy:
    name = "policy"

    def reset(self, env):
        pass

    def act(self, env):
        raise NotImplementedError


class Fixed(Policy):
    def __init__(self, value=1.0):
        self.value, self.name = value, f"fixed_{value:g}"

    def act(self, env):
        return np.full(env.K, self.value)


class RandomPolicy(Policy):
    name = "random"

    def act(self, env):
        return env._to_mult(env.rng.uniform(-1, 1, env.K))


class RuleBased(Policy):
    """Surge proportional to the demand/supply imbalance, like a classic heuristic surge."""

    def __init__(self, k=1.0, cap=3.0):
        self.k, self.cap, self.name = k, cap, f"rule_k{k:g}"

    def act(self, env):
        ratio = env.expected_demand() / np.maximum(env.idle, 1)
        return np.clip(1.0 + self.k * (ratio - 1.0), 1.0, self.cap)


class Myopic(Policy):
    """One-step lookahead using the true elasticity and the same reward as the RL agent.

    Optimistic reference: it knows the demand model, but ignores that today's price changes
    tomorrow's supply (driver repositioning and fleet entry).
    """
    name = "myopic_oracle"

    def __init__(self, grid=25):
        self.grid = grid

    def act(self, env):
        M = np.linspace(1.0, env.max_mult, self.grid)[:, None]          # (G, 1)
        lam, idle = env.expected_demand()[None], env.idle[None]         # (1, K)
        served = np.minimum(lam * M ** -env.eps, idle)                  # (G, K)
        value = (env.take * env.fare * M * served / env.rev_scale
                 - env.w["unmet"] * (lam - served) / env.arr_scale)
        return M[value.argmax(axis=0), 0]


class SB3Policy(Policy):
    def __init__(self, path, name="ppo"):
        from stable_baselines3 import PPO
        self.model, self.name = PPO.load(path), name

    def act(self, env):
        action, _ = self.model.predict(env._obs(), deterministic=True)
        return env._to_mult(action)
