import random
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import torch
import torch.nn as nn
import torch.nn.functional as F

import ale_py

# =====================================================
# Replay Buffer
# =====================================================

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        transitions = random.sample(
            self.buffer,
            batch_size
        )
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones
        )

    def size(self):
        return len(self.buffer)


# =====================================================
# CNN Q Network (Nature DQN)
# =====================================================
class ConvolutionalQnet(nn.Module):
    """
    输入：
        (4,84,84)

    输出：
        每个动作对应的Q值
    """

    def __init__(self,
                 action_dim,
                 in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            32,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=4,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=1
        )

        self.fc1 = nn.Linear(
            7 * 7 * 64,
            512
        )

        self.head = nn.Linear(
            512,
            action_dim
        )

    def forward(self, x):

        # 像素归一化
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.head(x)


# =====================================================
# DQN
# =====================================================

class DQN:

    def __init__(
            self,
            action_dim,
            learning_rate,
            gamma,
            epsilon,
            target_update,
            device):

        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

        # 在线网络
        self.q_net = ConvolutionalQnet(
            action_dim
        ).to(device)

        # 目标网络
        self.target_q_net = ConvolutionalQnet(
            action_dim
        ).to(device)

        self.target_q_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=learning_rate
        )

    # -------------------------------------------------

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.tensor(
            np.array(state),
            dtype=torch.float,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.q_net(state).argmax().item()
        return action

    # -------------------------------------------------

    def update(self, transition_dict):
        states = torch.tensor(
            np.array(transition_dict["states"]),
            dtype=torch.float,
            device=self.device
        )

        actions = torch.tensor(
            transition_dict["actions"],
            dtype=torch.long,
            device=self.device
        ).view(-1, 1)

        rewards = torch.tensor(
            transition_dict["rewards"],
            dtype=torch.float,
            device=self.device
        ).view(-1, 1)

        next_states = torch.tensor(
            np.array(transition_dict["next_states"]),
            dtype=torch.float,
            device=self.device
        )

        dones = torch.tensor(
            transition_dict["dones"],
            dtype=torch.float,
            device=self.device
        ).view(-1, 1)

        # 当前Q值
        q_values = self.q_net(states).gather(
            1,
            actions
        )

        # 下一状态Q值
        with torch.no_grad():
            max_next_q_values = self.target_q_net(
                next_states
            ).max(1)[0].view(-1, 1)
            q_targets = rewards + \
                        self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(
            q_values,
            q_targets
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )

        self.count += 1
# =====================================================
# Hyper Parameters
# =====================================================

learning_rate = 1e-4
num_episodes = 5000
gamma = 0.99
epsilon = 1.0          # 初始探索率
epsilon_min = 0.1
epsilon_decay = 1e-6
target_update = 1000
buffer_size = 100000
minimal_size = 50000
batch_size = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# =====================================================
# Atari Environment
# =====================================================

env_name = "ALE/Breakout-v5"

env = gym.make(
    env_name,
    render_mode=None,
    frameskip=1
)

# Atari预处理
env = AtariPreprocessing(
    env,
    frame_skip=4,
    screen_size=84,
    grayscale_obs=True,
    grayscale_newaxis=False,
    scale_obs=False
)

# 连续4帧作为一个状态
env = FrameStackObservation(
    env,
    stack_size=4
)


# =====================================================
# Environment Information
# =====================================================

action_dim = env.action_space.n

print("Action Number :", action_dim)

print("Observation Shape :", env.observation_space.shape)

# 输出应该类似：
# (4,84,84)


# =====================================================
# Replay Buffer
# =====================================================

replay_buffer = ReplayBuffer(buffer_size)


# =====================================================
# Agent
# =====================================================

agent = DQN(
    action_dim=action_dim,
    learning_rate=learning_rate,
    gamma=gamma,
    epsilon=epsilon,
    target_update=target_update,
    device=device
)


# =====================================================
# Training Recorder
# =====================================================

return_list = []

loss_list = []

total_steps = 0


# =====================================================
# Start Training
# =====================================================
# =====================================================
# Training Loop
# =====================================================

for i in range(10):

    with tqdm(
            total=num_episodes // 10,
            desc=f"Iteration {i + 1}") as pbar:

        for i_episode in range(num_episodes // 10):

            # 每个Episode使用不同随机种子
            state, info = env.reset(
                seed=i * (num_episodes // 10) + i_episode
            )

            done = False

            episode_return = 0

            while not done:

                total_steps += 1

                # -------------------------------
                # Choose Action
                # -------------------------------
                action = agent.take_action(state)

                # -------------------------------
                # Environment Step
                # -------------------------------
                next_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                # -------------------------------
                # Store Transition
                # -------------------------------
                replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                state = next_state

                episode_return += reward

                # -------------------------------
                # Network Update
                # -------------------------------
                if replay_buffer.size() >= minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "rewards": b_r,
                        "next_states": b_ns,
                        "dones": b_d
                    }
                    agent.update(transition_dict)
                    # epsilon衰减
                    if agent.epsilon > epsilon_min:
                        agent.epsilon -= epsilon_decay
                        if agent.epsilon < epsilon_min:
                            agent.epsilon = epsilon_min
            # -----------------------------------
            # Record Return
            # -----------------------------------
            return_list.append(
                episode_return
            )

            # -----------------------------------
            # Show Training Info
            # -----------------------------------
            if (i_episode + 1) % 10 == 0:
                mean_return = np.mean(
                    return_list[-10:]
                )

                pbar.set_postfix({
                    "Episode":
                    i * (num_episodes // 10)
                    + i_episode + 1,
                    "Return":
                    "%.2f" % mean_return,
                    "Buffer":
                    replay_buffer.size(),
                    "Epsilon":
                    "%.3f" % agent.epsilon
                })
            pbar.update(1)
env.close()

def moving_average(data, window_size=20):

    if len(data) < window_size:
        return data

    weights = np.ones(window_size) / window_size

    return np.convolve(
        data,
        weights,
        mode="valid"
    )

episodes = np.arange(len(return_list))

plt.figure(figsize=(10, 5))

plt.plot(
    episodes,
    return_list,
    label="Return"
)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN on Atari Breakout")
plt.grid(True)
plt.legend()
plt.show()

mv_return = moving_average(
    return_list,
    window_size=20
)

plt.figure(figsize=(10, 5))
plt.plot(
    np.arange(len(mv_return)),
    mv_return,
    linewidth=2,
    label="Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Moving Average Return")
plt.grid(True)
plt.legend()
plt.show()
