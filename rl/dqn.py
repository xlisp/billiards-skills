"""
DQN（Double DQN 变体）。

- QNetwork: 简单 MLP，输入是平铺的球状态，输出每个离散动作的 Q 值。
- ReplayBuffer: numpy 实现，按 transition 存储。
- DQNAgent: ε-greedy 选动作，TD 目标用 target network。
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, o, a, r, no, d):
        i = self.idx
        self.obs[i] = o
        self.actions[i] = a
        self.rewards[i] = r
        self.next_obs[i] = no
        self.dones[i] = float(d)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx])


class DQNAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 device: str = "cpu",
                 lr: float = 3e-4,
                 gamma: float = 0.97,
                 buffer_size: int = 50_000,
                 batch_size: int = 128,
                 target_update_steps: int = 1000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 30_000):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.q = QNetwork(obs_dim, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.train_steps = 0

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        q = self.q(x)
        return int(q.argmax(dim=1).item())

    def learn(self) -> float | None:
        if self.buffer.size < self.batch_size:
            return None

        o, a, r, no, d = self.buffer.sample(self.batch_size)
        o = torch.from_numpy(o).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        no = torch.from_numpy(no).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        q_pred = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: 用在线网络选动作，用 target 网络估值
            next_actions = self.q(no).argmax(dim=1, keepdim=True)
            next_q = self.q_target(no).gather(1, next_actions).squeeze(1)
            target = r + self.gamma * next_q * (1.0 - d)

        loss = F.smooth_l1_loss(q_pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(),
                    "q_target": self.q_target.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
