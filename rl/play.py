"""
加载训练好的 DQN，可视化它打球。

用法：
    python play.py --ckpt ckpt/dqn_latest.pt --balls 3 --episodes 5
    python play.py --random   # 不加载模型，看随机策略对照
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from billiards_env import BilliardsEnv
from dqn import DQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt/dqn_latest.pt")
    p.add_argument("--balls", type=int, default=3)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-render", action="store_true", help="只跑数据，不开窗口")
    p.add_argument("--random", action="store_true", help="不加载模型，随机策略")
    return p.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.no_render else "human"
    env = BilliardsEnv(n_object_balls=args.balls, render_mode=render_mode, seed=args.seed)

    agent = None
    if not args.random:
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            device=args.device,
        )
        agent.load(args.ckpt)
        print(f"loaded {args.ckpt}")

    totals = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_pocket = 0
        max_streak = 0
        env.render()
        while True:
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, eps=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_pocket += info["pocketed"]
            max_streak = max(max_streak, info["streak"])
            env.render()
            if terminated or truncated:
                break
        totals.append((ep_reward, ep_pocket, max_streak, info["shots"]))
        print(f"ep {ep+1}: reward={ep_reward:+.2f}  pocketed={ep_pocket}/{args.balls}  "
              f"max_streak={max_streak}  shots={info['shots']}")

    if totals:
        arr = np.array(totals)
        print(f"\nmean over {len(totals)} episodes: "
              f"reward={arr[:,0].mean():+.2f}  pocketed={arr[:,1].mean():.2f}  "
              f"streak={arr[:,2].mean():.2f}  shots={arr[:,3].mean():.1f}")

    env.close()


if __name__ == "__main__":
    main()
