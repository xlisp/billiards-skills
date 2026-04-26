"""
简化的桌球 Gymnasium 环境。

物理依据见 ../docs/physics.md：
- §二 动量守恒 → 球-球弹性碰撞
- §三 角动量与摩擦 → 简化为线性减速（这里不建模旋转）
- §五 路径积分 → 欧拉积分到所有球停下
- §六 库边反射 → 法向反弹 + 能量损失

动作: Discrete(N_ANGLES * N_FORCES)，离散化的 (出杆角度, 出杆力度)。
观测: 每个球的 (x_norm, y_norm, alive) 拼成一维向量。
奖励: 进球数 × 连击倍率 - 每杆代价 - 白球进袋惩罚 + 清台奖励。
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BilliardsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # 物理常量（来自 docs/physics.md）
    BALL_RADIUS = 0.02858
    BALL_MASS = 0.17
    TABLE_W = 2.24
    TABLE_H = 1.12
    POCKET_RADIUS = 0.060
    MU = 0.20            # 桌呢摩擦系数
    G = 9.8
    RESTITUTION = 0.96   # 球-球弹性恢复
    CUSHION_E = 0.70     # 库边反弹

    # 动作离散化
    N_ANGLES = 36
    N_FORCES = 5
    FORCE_MIN = 1.5
    FORCE_MAX = 5.5      # m/s 母球初速度

    DT = 0.005
    MAX_SIM_T = 8.0
    MAX_SHOTS = 25

    def __init__(self, n_object_balls: int = 3, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.n_object_balls = n_object_balls
        self.n_balls = n_object_balls + 1
        self.render_mode = render_mode

        self.pockets = np.array([
            [0.0,            0.0],
            [self.TABLE_W/2, 0.0],
            [self.TABLE_W,   0.0],
            [0.0,            self.TABLE_H],
            [self.TABLE_W/2, self.TABLE_H],
            [self.TABLE_W,   self.TABLE_H],
        ], dtype=np.float64)

        self.action_space = spaces.Discrete(self.N_ANGLES * self.N_FORCES)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3 * self.n_balls,), dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._screen = None
        self._traj_buffer: list[np.ndarray] = []  # 记录轨迹用于渲染

    # -----------------------------------------------------------------
    # Gym API
    # -----------------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cue = np.array([self.TABLE_W * 0.25, self.TABLE_H * 0.5])
        positions = [cue]
        for _ in range(self.n_object_balls):
            for _ in range(100):
                p = np.array([
                    self._rng.uniform(self.TABLE_W * 0.45, self.TABLE_W * 0.95),
                    self._rng.uniform(self.TABLE_H * 0.10, self.TABLE_H * 0.90),
                ])
                if all(np.linalg.norm(p - q) > 3 * self.BALL_RADIUS for q in positions):
                    positions.append(p)
                    break
            else:
                positions.append(p)

        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.alive = np.ones(self.n_balls, dtype=bool)

        self.shots = 0
        self.streak = 0
        self.total_pocketed = 0
        self._traj_buffer = [self.positions.copy()]

        return self._get_obs(), {}

    def step(self, action: int):
        angle, force = self._decode_action(action)

        if self.alive[0]:
            self.velocities[0] = force * np.array([np.cos(angle), np.sin(angle)])

        pocketed_this_shot = self._simulate()
        cue_pocketed = not self.alive[0]

        # 奖励：连击倍率（README §九 连续进攻 / 球形选择）
        reward = 0.0
        if pocketed_this_shot > 0:
            self.streak += 1
            multiplier = 1.0 + 0.5 * (self.streak - 1)
            reward += pocketed_this_shot * multiplier
        else:
            self.streak = 0

        reward -= 0.1                       # 每杆代价 → 鼓励"越快越好"
        if cue_pocketed:
            reward -= 3.0                   # 白球进袋 = 犯规

        self.shots += 1
        terminated = False

        if not self.alive[1:].any():
            reward += 5.0                   # 清台奖励
            terminated = True
        if cue_pocketed:
            terminated = True

        truncated = self.shots >= self.MAX_SHOTS

        info = {
            "pocketed": pocketed_this_shot,
            "streak": self.streak,
            "shots": self.shots,
            "remaining": int(self.alive[1:].sum()),
            "cue_pocketed": cue_pocketed,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # -----------------------------------------------------------------
    # 内部物理
    # -----------------------------------------------------------------
    def _decode_action(self, action: int) -> tuple[float, float]:
        ang_idx = action // self.N_FORCES
        force_idx = action % self.N_FORCES
        angle = (ang_idx / self.N_ANGLES) * 2.0 * np.pi
        if self.N_FORCES > 1:
            force = self.FORCE_MIN + (force_idx / (self.N_FORCES - 1)) * (self.FORCE_MAX - self.FORCE_MIN)
        else:
            force = self.FORCE_MAX
        return angle, force

    def _simulate(self) -> int:
        pocketed = 0
        n_steps = int(self.MAX_SIM_T / self.DT)
        sample_every = 8  # 每 8 步采样一次轨迹用于渲染

        for step_idx in range(n_steps):
            if not self.alive.any():
                break
            speeds = np.linalg.norm(self.velocities, axis=1)
            if speeds.max() < 0.05:
                break

            # 1. 摩擦减速
            for i in range(self.n_balls):
                if not self.alive[i]:
                    continue
                spd = speeds[i]
                if spd > 1e-6:
                    decel = self.MU * self.G * self.DT
                    new_spd = max(0.0, spd - decel)
                    self.velocities[i] *= (new_spd / spd) if new_spd > 0 else 0.0

            # 2. 位置更新
            self.positions += self.velocities * self.DT

            # 3. 库边反弹
            for i in range(self.n_balls):
                if not self.alive[i]:
                    continue
                p, v = self.positions[i], self.velocities[i]
                if p[0] < self.BALL_RADIUS:
                    p[0] = self.BALL_RADIUS
                    v[0] = -v[0] * self.CUSHION_E
                elif p[0] > self.TABLE_W - self.BALL_RADIUS:
                    p[0] = self.TABLE_W - self.BALL_RADIUS
                    v[0] = -v[0] * self.CUSHION_E
                if p[1] < self.BALL_RADIUS:
                    p[1] = self.BALL_RADIUS
                    v[1] = -v[1] * self.CUSHION_E
                elif p[1] > self.TABLE_H - self.BALL_RADIUS:
                    p[1] = self.TABLE_H - self.BALL_RADIUS
                    v[1] = -v[1] * self.CUSHION_E

            # 4. 球-球弹性碰撞（沿法线方向交换动量，对应 physics.md §二）
            for i in range(self.n_balls):
                if not self.alive[i]:
                    continue
                for j in range(i + 1, self.n_balls):
                    if not self.alive[j]:
                        continue
                    d = self.positions[j] - self.positions[i]
                    dist = np.linalg.norm(d)
                    if dist < 2 * self.BALL_RADIUS and dist > 1e-9:
                        normal = d / dist
                        v_rel = self.velocities[j] - self.velocities[i]
                        v_rel_n = v_rel @ normal
                        if v_rel_n < 0:
                            impulse = self.RESTITUTION * v_rel_n
                            self.velocities[i] += impulse * normal
                            self.velocities[j] -= impulse * normal
                        # 分离重叠（防止粘连）
                        overlap = 2 * self.BALL_RADIUS - dist
                        self.positions[i] -= normal * (overlap * 0.5)
                        self.positions[j] += normal * (overlap * 0.5)

            # 5. 进袋判定
            for i in range(self.n_balls):
                if not self.alive[i]:
                    continue
                for pk in self.pockets:
                    if np.linalg.norm(self.positions[i] - pk) < self.POCKET_RADIUS:
                        self.alive[i] = False
                        self.velocities[i] = 0.0
                        if i != 0:
                            pocketed += 1
                            self.total_pocketed += 1
                        break

            if step_idx % sample_every == 0:
                self._traj_buffer.append(self.positions.copy())

        self.velocities[:] = 0.0
        self._traj_buffer.append(self.positions.copy())
        return pocketed

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(3 * self.n_balls, dtype=np.float32)
        for i in range(self.n_balls):
            if self.alive[i]:
                obs[3*i + 0] = (self.positions[i, 0] / self.TABLE_W) * 2.0 - 1.0
                obs[3*i + 1] = (self.positions[i, 1] / self.TABLE_H) * 2.0 - 1.0
                obs[3*i + 2] = 1.0
            else:
                obs[3*i + 0] = 0.0
                obs[3*i + 1] = 0.0
                obs[3*i + 2] = 0.0
        return obs

    # -----------------------------------------------------------------
    # 渲染（pygame）
    # -----------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None
        try:
            import pygame
        except ImportError:
            print("pygame not installed; skipping render. pip install pygame")
            return None

        scale = 300                          # 米 → 像素
        W = int(self.TABLE_W * scale)
        H = int(self.TABLE_H * scale)

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Billiards DQN")
            self._clock = pygame.time.Clock()

        for frame_pos in self._traj_buffer:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self._screen = None
                    return
            self._screen.fill((20, 100, 50))   # 桌呢绿
            for pk in self.pockets:
                pygame.draw.circle(self._screen, (10, 10, 10),
                                   (int(pk[0]*scale), int(pk[1]*scale)),
                                   int(self.POCKET_RADIUS * scale))
            for i in range(self.n_balls):
                if i < len(frame_pos) and self.alive[i] or (frame_pos is not self._traj_buffer[-1]):
                    # 被击落的球在轨迹回放中也显示，最终帧才隐藏
                    pos = frame_pos[i]
                    color = (240, 240, 240) if i == 0 else (220, 50, 50)
                    pygame.draw.circle(self._screen, color,
                                       (int(pos[0]*scale), int(pos[1]*scale)),
                                       int(self.BALL_RADIUS * scale))
            pygame.display.flip()
            self._clock.tick(60)

        self._traj_buffer = [self.positions.copy()]
