"""Benchmark all policies on identical seeds across scenarios.

    python -m pricing_rl.benchmark --episodes 30
    python -m pricing_rl.benchmark --episodes 30 --ppo models/ppo.zip
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import Fixed, Myopic, RandomPolicy, RuleBased, SB3Policy
from .env import SCENARIOS, make_env

TRAIN_SEEDS = range(0, 20)       # used only to tune baseline hyperparameters
TEST_SEED0 = 10_000              # test seeds never overlap tuning/training


def run_episode(env, policy, seed):
    env.reset(seed=seed)
    policy.reset(env)
    total, done = 0.0, False
    while not done:
        m = policy.act(env)
        _, r, term, trunc, info = env.step(env.mult_to_action(m))
        total += r
        done = term or trunc
    return {"reward": total, **info["episode"]}


def evaluate(policy, scenario, seeds, profile):
    env = make_env(scenario, profile)
    return pd.DataFrame([run_episode(env, policy, s) for s in seeds])


def tune(profile):
    """Pick the best fixed multiplier and rule-based gain on tuning seeds (weekday scenario)."""
    def score(p):
        return evaluate(p, "weekday", TRAIN_SEEDS, profile)["reward"].mean()
    fixed = max((Fixed(v) for v in (1.0, 1.25, 1.5, 1.75, 2.0)), key=score)
    rule = max((RuleBased(k) for k in (0.25, 0.5, 1.0, 2.0)), key=score)
    return fixed, rule


def ci95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="data/profile.npz")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--ppo", default=None, help="path to a trained SB3 PPO model")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    best_fixed, best_rule = tune(args.profile)
    best_fixed.name = f"fixed_best({best_fixed.value:g})"
    best_rule.name = f"rule_based(k={best_rule.k:g})"
    policies = [Fixed(1.0), best_fixed, best_rule, Myopic(), RandomPolicy()]
    if args.ppo:
        policies.append(SB3Policy(args.ppo))

    seeds = range(TEST_SEED0, TEST_SEED0 + args.episodes)
    rows = []
    for scenario in SCENARIOS:
        for pol in policies:
            df = evaluate(pol, scenario, seeds, args.profile)
            df.insert(0, "seed", list(seeds))
            df.insert(0, "policy", pol.name)
            df.insert(0, "scenario", scenario)
            rows.append(df)
    results = pd.concat(rows, ignore_index=True)

    Path(args.out).mkdir(exist_ok=True)
    results.to_csv(f"{args.out}/episodes.csv", index=False)

    metrics = ["reward", "revenue", "service_rate", "mean_mult", "max_mult", "volatility",
               "zone_service_gap"]
    g = results.groupby(["scenario", "policy"], sort=False)[metrics]
    summary = g.mean().round(3).astype(str) + " ±" + g.agg(ci95).round(3).astype(str)
    summary.to_csv(f"{args.out}/summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(summary)


if __name__ == "__main__":
    main()
