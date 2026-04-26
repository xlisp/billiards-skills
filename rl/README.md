# DQN 桌球 RL

让 DQN 自己学会打桌球——目标：**连续进球越快越多 reward 越高**。

环境基于 [`../docs/physics.md`](../docs/physics.md) 的物理（动量守恒 / 摩擦衰减 / 库边反射 / 进袋判定），奖励对应 [`../README.md`](../README.md) §4 §5 §9 的实战要义（效率、连击、容错）。

## 文件

| 文件 | 作用 |
|---|---|
| `billiards_env.py` | Gymnasium 环境：离散动作 `(角度, 力度)`，观测 = 球状态 |
| `dqn.py`           | Double-DQN：QNetwork + ReplayBuffer + DQNAgent |
| `train.py`         | 训练循环 + checkpoint |
| `play.py`          | 加载模型，pygame 可视化 |

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
cd rl
python train.py --episodes 5000 --balls 3
```

常用参数：

```
--balls 3              # 目标球数（cue ball 之外）
--episodes 5000
--device cpu           # 或 cuda / mps
--lr 3e-4
--gamma 0.97
--eps-decay 30000      # ε 线性从 1.0 衰减到 0.05 经历的环境步数
--save-every 500
```

训练日志示例：

```
[ep   200] R̄=+0.31  pocket̄=0.65/3  streak̄=0.40  shots̄=18.2  clear%= 4  ε=0.83
[ep  1000] R̄=+1.85  pocket̄=1.40/3  streak̄=1.10  shots̄=15.1  clear%=22  ε=0.51
[ep  3000] R̄=+4.10  pocket̄=2.40/3  streak̄=1.90  shots̄=11.5  clear%=58  ε=0.10
[ep  5000] R̄=+6.30  pocket̄=2.85/3  streak̄=2.55  shots̄= 8.7  clear%=85  ε=0.05
```

`R̄` 升高、`shots̄` 下降、`streak̄` 上升 = AI 越来越像高手。

## 可视化

```bash
python play.py --ckpt ckpt/dqn_latest.pt --episodes 5
python play.py --random   # 随机策略对照组
```

会弹出 pygame 窗口，回放每一杆的轨迹。

## 设计要点

### 动作空间
- `Discrete(N_ANGLES * N_FORCES)` = 36 × 5 = 180
- 不建模旋转（高低杆 / 加塞）。spin 可以后续作为额外的离散维度加入。

### 观测空间
- `(n_balls, 3)` 拍平：每个球 `(x_norm, y_norm, alive)`
- 进袋的球位置归零 + alive=0

### 奖励（核心：实现"连续进球越快越多越好"）

| 事件 | reward |
|---|---|
| 进 1 颗目标球 | `+1.0 × (1 + 0.5×(streak-1))` |
| 一杆进 N 颗 | 上式 × N |
| 每出一杆 | `-0.1`（鼓励"快"）|
| 母球进袋 | `-3.0`（README §7 犯规惩罚）|
| 全清台 | `+5.0` |
| 失败一杆（没进球）| streak 清零 |

`streak` = 连续多少杆都进了球。从 README §9 来：高手单杆数高 = 连续不断杆。

### 物理简化
对应 `docs/physics.md` 的实现：
- §二 动量守恒 → 球-球弹性碰撞（沿法线方向交换动量）
- §三 摩擦 → 简化为线性减速 `μg`，**未建模旋转**（DQN 任务足够）
- §五 路径积分 → 欧拉积分到所有球停下
- §六 库边反射 → 法向反弹 + 切向能量损失

## 后续可改进

1. **加入旋转**：把 `hit_offset` 也作为离散动作维度，让 agent 学会用高低杆控制走位（README §1.2）。
2. **更稠密奖励**：靠近袋口时给 shaping reward，加速早期学习。
3. **Frame-stack**：拍多帧观测捕捉走位意图。
4. **PPO / SAC**：连续动作空间会更精细，但 DQN 已经足以让 agent 学到"瞄袋口"的几何直觉。
5. **课程学习**：先 1 球，逐步加到 5 球。
