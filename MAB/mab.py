import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """ 模拟多臂老虎机，每个臂的奖励服从伯努利分布（p的概率奖励为1， 1-p的概率奖励为0）， K表示臂的数量 """
    def __init__(self, K):
        self.K = K
        # 初始化多臂老虎机，随机生成每个臂的获奖概率
        self.probs = np.random.uniform(size = K)
        # 记录最优臂和最优概率
        self.best_arm = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_arm]
    def step(self, action):
        """ 根据选择的臂返回奖励。
        生成一个 [0, 1) 上的均匀随机数，它小于 p 的概率恰好就是 p """
        if np.random.rand() < self.probs[action]:
            return 1
        else:
            return 0


class Solver:
    
    def __init__(self, bandit):
        self.bandit = bandit
        # 记录每个臂被选择的次数
        self.cnt = np.zeros(bandit.K)
        # 当前的累积懊悔
        self.regret = 0
        # 记录每个时间步的动作
        self.actions = []
        # 记录每个时间步的累积懊悔
        self.regrets = []
    
    def update_regret(self, action):
        """ 更新累积懊悔 """
        self.regret += self.bandit.best_prob - self.bandit.probs[action]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """ 运行一个时间步，选择一个臂并更新累积懊悔 """
        raise NotImplementedError

    def run(self, num_steps):
        """ 运行多臂老虎机 num_steps 步 """
        for _ in range(num_steps):
            action = self.run_one_step()
            self.actions.append(action)
            self.cnt[action] += 1
            self.update_regret(action)


# 平衡探索和利用的策略
# 1. Epsilon-Greedy 算法：以 ε 的概率随机选择一个臂（探索），以 1-ε 的概率选择当前估计最优的臂（利用）。
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon = 0.1, init_prob = 1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化每个臂的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)
    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            # 探索：随机选择一个臂
            action = np.random.choice(self.bandit.K)
        else:
            action = np.argmax(self.estimates)
        # 获取奖励
        r = self.bandit.step(action)
        # 更新期望奖励估值，使用增量更新公式
        self.estimates[action] += (r - self.estimates[action]) / (self.cnt[action] + 1)
        return action

# 2.epsilon-greedy 算法, epsilon 随时间衰减, 衰减形式为反比例衰减
class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob = 1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        # 初始化每个臂的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_steps = 0
    def run_one_step(self):
        # 计算当前的epsilon值，随着时间步数增加而衰减
        self.total_steps += 1
        epsilon = 1 / self.total_steps
        if np.random.rand() < epsilon:
            # 探索：随机选择一个臂
            action = np.random.choice(self.bandit.K)
        else:
            action = np.argmax(self.estimates)
        # 获取奖励
        r = self.bandit.step(action)
        # 更新期望奖励估值
        self.estimates[action] += (r - self.estimates[action]) / (self.cnt[action] + 1)
        return action
    

# 3. Upper Confidence Bound (UCB) 算法：在选择臂时考虑当前估计的奖励和不确定性，选择具有最高上置信界的臂。
class UCB(Solver):
    def __init__(self, bandit, coef, init_prob = 1.0):
        super(UCB, self).__init__(bandit)
        # 初始化每个臂的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)
        # UCB算法中的coef参数控制探索程度，coef越大，算法越倾向于探索
        self.coef = coef
        self.total_steps = 0
    
    def run_one_step(self):
        self.total_steps += 1
        # 计算每个臂的上置信界 设置了 p = 1/t 可选设计 p = 1/t^2; p = 1/t^4 等衰减形式
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_steps) / (2 * (self.cnt + 1))) # 加1避免除以0
        action = np.argmax(ucb)
        # 获取奖励
        r = self.bandit.step(action)
        # 更新期望奖励估值
        self.estimates[action] += (r - self.estimates[action]) / (self.cnt[action] + 1)
        return action
    

# 4. Thompson Sampling 算法：为每个臂维护一个 Beta 分布的后验分布，根据当前的后验分布采样来选择臂。
class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        # 初始化每个臂的 Beta 分布参数，alpha 和 beta
        self.alpha = np.ones(bandit.K)
        self.beta = np.ones(bandit.K)
    def run_one_step(self):
        # 按照当前的 Beta 分布参数为每个臂采样一个值
        samples = np.random.beta(self.alpha, self.beta)
        action = np.argmax(samples)
        # 获取奖励
        r = self.bandit.step(action)
        # 更新 Beta 分布参数
        self.alpha[action] += r
        self.beta[action] += 1 - r
        return action


# 绘制结果
def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        plt.plot(solver.regrets, label = solver_names[idx])
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret of Different Bandit Algorithms')
    plt.legend()
    plt.show()


# 运行实验
bandit = BernoulliBandit(K = 10)
num_steps = 5000
solvers = [
    EpsilonGreedy(bandit, epsilon = 0.1),
    DecayingEpsilonGreedy(bandit),
    UCB(bandit, coef = 1),
    ThompsonSampling(bandit)
]
solver_names = ['Epsilon-Greedy', 'Decaying Epsilon-Greedy', 'UCB', 'Thompson Sampling']
for solver in solvers:
    solver.run(num_steps)
plot_results(solvers, solver_names)

solvers_epsilon = [EpsilonGreedy(bandit, epsilon = 0.1), EpsilonGreedy(bandit, epsilon = 0.01), EpsilonGreedy(bandit, epsilon = 0.001)]
sets_epsilon_names = ['Epsilon-Greedy (ε=0.1)', 'Epsilon-Greedy (ε=0.01)', 'Epsilon-Greedy (ε=0.001)']
for solver in solvers_epsilon:
    solver.run(num_steps)
plot_results(solvers_epsilon, sets_epsilon_names)