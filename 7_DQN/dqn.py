import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """DQN"""

    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device):

        self.action_dim = action_dim

        self.q_net = Qnet(
            state_dim,
            hidden_dim,
            action_dim).to(device)

        self.target_q_net = Qnet(
            state_dim,
            hidden_dim,
            action_dim).to(device)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.tensor([state], dtype=torch.float, device=self.device)

        return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.float,
            device=self.device)
        actions = torch.tensor(
            transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype=torch.float,
            device=self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        self.count += 1
if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10

    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    env.reset(seed=0)
    torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(
        state_dim,
        hidden_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device)

    return_list = []

    for i in range(10):
        with tqdm(total=num_episodes // 10,
                desc=f"Iteration {i}") as pbar:
            for i_episode in range(num_episodes // 10):
                state, info = env.reset(seed=0)
                done = False
                episode_return = 0
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    replay_buffer.add(
                        state,
                        action,
                        reward,
                        next_state,
                        done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }

                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        "episode":
                        num_episodes // 10 * i + i_episode + 1,
                        "return":
                        "%.3f" % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()

    episodes_list = list(range(len(return_list)))

    plt.figure(figsize=(8, 5))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"DQN on {env_name}")
    plt.grid(True)
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes_list, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"DQN on {env_name}")
    plt.grid(True)
    plt.show()