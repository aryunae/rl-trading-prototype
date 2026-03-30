#!/usr/bin/env python
# src/train.py

import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trading_env import TradingEnvironment
from src.dqn_agent import DDQNAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDQN agent for trading")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--trading_days", type=int, default=252, help="Episode length in trading days")
    parser.add_argument("--trading_cost", type=float, default=1e-3, help="Trading cost in basis points")
    parser.add_argument("--time_cost", type=float, default=1e-4, help="Time cost per step")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=250, help="Linear decay steps")
    parser.add_argument("--epsilon_exp_decay", type=float, default=0.99, help="Exponential decay factor")
    parser.add_argument("--replay_capacity", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--architecture", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes")
    parser.add_argument("--l2_reg", type=float, default=1e-6, help="L2 regularization")
    parser.add_argument("--tau", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Minibatch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()

def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create environment
    env = TradingEnvironment(
        trading_days=args.trading_days,
        trading_cost_bps=args.trading_cost,
        time_cost_bps=args.time_cost,
        ticker=args.ticker
    )
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    max_episode_steps = args.trading_days

    # Create agent
    agent = DDQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_exponential_decay=args.epsilon_exp_decay,
        replay_capacity=args.replay_capacity,
        architecture=tuple(args.architecture),
        l2_reg=args.l2_reg,
        tau=args.tau,
        batch_size=args.batch_size
    )

    # Prepare results storage
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    navs = []
    market_navs = []
    diffs = []

    # Training loop
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < max_episode_steps:
            action = agent.epsilon_greedy_policy(state.reshape(1, -1))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.memorize_transition(state, action, reward, next_state, 0.0 if done else 1.0)
            if agent.train:
                agent.experience_replay()
            state = next_state
            episode_reward += reward
            step += 1

        # Record episode results
        result = env.simulator.result()
        final = result.iloc[-1]
        nav = final.nav * (1 + final.strategy_return)
        market_nav = final.market_nav
        navs.append(nav)
        market_navs.append(market_nav)
        diffs.append(nav - market_nav)

        if episode % 10 == 0:
            win_ratio = np.sum([d > 0 for d in diffs[-100:]]) / min(len(diffs), 100)
            print(f"Episode {episode:4d}: Agent NAV = {nav:.4f}, Market NAV = {market_nav:.4f}, "
                  f"Diff = {nav-market_nav:.4f}, Win Ratio (100) = {win_ratio:.2%}, ε = {agent.epsilon:.4f}")

    env.close()

    # Save results
    results_df = pd.DataFrame({
        'Episode': list(range(1, args.episodes + 1)),
        'Agent': navs,
        'Market': market_navs,
        'Difference': diffs
    })
    results_df.to_csv(results_path / 'results.csv', index=False)

    # Plot results
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    df1 = (results_df[['Agent', 'Market']].sub(1).rolling(100).mean())
    df1.plot(ax=axes[0], title='Annual Returns (Moving Average)', lw=1)
    df2 = (results_df['Difference'] > 0).rolling(100).mean()
    df2.plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)')
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    axes[1].axhline(0.5, ls='--', c='k', lw=1)
    fig.tight_layout()
    fig.savefig(results_path / 'performance.png', dpi=300)
    plt.close(fig)

    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)