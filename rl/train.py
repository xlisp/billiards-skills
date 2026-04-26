"""
DQN 训练入口。

用法：
    python train.py --episodes 5000 --balls 3 --device cpu

每隔 --log-every 个 episode 打印一次：平均奖励 / 平均进球 / 平均连击 / ε。
按 --save-every 保存 checkpoint 到 ckpt/ 目录。
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from billiards_env import BilliardsEnv
from dqn import DQNAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--balls", type=int, default=3, help="目标球数量")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str, default="ckpt")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--eps-decay", type=int, default=30_000, help="ε 线性衰减经历的环境步数")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = BilliardsEnv(n_object_balls=args.balls, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eps_decay_steps=args.eps_decay,
    )

    rewards_window = deque(maxlen=args.log_every)
    pocket_window = deque(maxlen=args.log_every)
    streak_window = deque(maxlen=args.log_every)
    shots_window = deque(maxlen=args.log_every)
    cleared_window = deque(maxlen=args.log_every)

    global_step = 0
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_pocket = 0
        ep_max_streak = 0

        while True:
            eps = agent.epsilon(global_step)
            action = agent.act(obs, eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, terminated)
            agent.learn()

            obs = next_obs
            ep_reward += reward
            ep_pocket += info["pocketed"]
            ep_max_streak = max(ep_max_streak, info["streak"])
            global_step += 1

            if done:
                cleared = (info["remaining"] == 0 and not info["cue_pocketed"])
                break

        rewards_window.append(ep_reward)
        pocket_window.append(ep_pocket)
        streak_window.append(ep_max_streak)
        shots_window.append(info["shots"])
        cleared_window.append(1.0 if cleared else 0.0)

        if ep % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[ep {ep:5d}] "
                  f"R̄={np.mean(rewards_window):+6.2f}  "
                  f"pocket̄={np.mean(pocket_window):4.2f}/{args.balls}  "
                  f"streak̄={np.mean(streak_window):4.2f}  "
                  f"shots̄={np.mean(shots_window):4.1f}  "
                  f"clear%={100*np.mean(cleared_window):4.0f}  "
                  f"ε={eps:.3f}  "
                  f"steps={global_step}  "
                  f"({elapsed:.0f}s)")

        if ep % args.save_every == 0:
            path = os.path.join(args.ckpt_dir, f"dqn_ep{ep}.pt")
            agent.save(path)
            latest = os.path.join(args.ckpt_dir, "dqn_latest.pt")
            agent.save(latest)
            print(f"  → saved {path}")

    final = os.path.join(args.ckpt_dir, "dqn_final.pt")
    agent.save(final)
    print(f"done. final ckpt: {final}")


if __name__ == "__main__":
    main()
