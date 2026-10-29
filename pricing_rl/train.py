"""Train a PPO pricing agent on a mix of market conditions.

    python -m pricing_rl.train --steps 1000000
Then evaluate against the baselines with:
    python -m pricing_rl.benchmark --ppo models/ppo.zip
"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .env import MixedScenarioEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="data/profile.npz")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo.zip")
    args = ap.parse_args()

    # every episode draws a random day of week plus (at most) one stress: driver shortage,
    # price-sensitive riders, or a local demand spike. Elasticity is also jittered.
    env = Monitor(MixedScenarioEnv(args.profile, elasticity_jitter=0.3))
    model = PPO("MlpPolicy", env, n_steps=960, batch_size=240, gamma=0.97, learning_rate=3e-4,
                ent_coef=0.005,  # small bonus for exploring instead of locking onto one price
                policy_kwargs=dict(net_arch=[128, 128]), seed=args.seed, verbose=1)
    model.learn(args.steps)
    Path(args.out).parent.mkdir(exist_ok=True)
    model.save(args.out)


if __name__ == "__main__":
    main()