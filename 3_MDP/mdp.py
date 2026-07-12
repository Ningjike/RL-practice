import numpy as np
from mrp import compute_value

np.random.seed(0)

# 状态集合
S = ["s1", "s2", "s3", "s4", "s5"]
# 动作集合
A = ["保持s1","前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1
}
gamma = 0.5  # 折扣因子

# 定义马尔可夫决策过程
MDP = (S, A, P, R, gamma)

# 策略1
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}

def marginalize_mdp_to_mrp(MDP, Pi):
    """
    将 MDP 在给定策略 Pi 下边缘化，转化为 MRP
    奖励: r(s) = Σ_a π(a|s) · r(s,a)
    转移: P(s'|s) = Σ_a π(a|s) · P(s'|s,a)
    """
    S, A, P, R, gamma = MDP
    n = len(S)
    state_idx = {s: i for i, s in enumerate(S)}

    # 初始化 MRP 的转移矩阵和奖励向量
    P_mrp = np.zeros((n, n))
    R_mrp = np.zeros(n)

    for s in S:
        i = state_idx[s]
        # 检查该状态是否有策略定义的动作
        has_action = any(f"{s}-{a}" in Pi for a in A)

        if not has_action:
            P_mrp[i, i] = 1.0
            continue

        for a in A:
            key_pi = f"{s}-{a}"
            if key_pi not in Pi:
                continue
            pi_a = Pi[key_pi]

            # 奖励边际化: r(s) += π(a|s) · r(s,a)
            key_r = f"{s}-{a}"
            if key_r in R:
                R_mrp[i] += pi_a * R[key_r]

            # 转移边际化: P(s'|s) += π(a|s) · P(s'|s,a)
            for s_prime in S:
                key_p = f"{s}-{a}-{s_prime}"
                if key_p in P:
                    j = state_idx[s_prime]
                    P_mrp[i, j] += pi_a * P[key_p]

    return P_mrp, R_mrp


def compute_action_value(MDP, V):
    """
    根据状态价值函数 V 计算动作价值函数 Q
    Q(s,a) = r(s,a) + γ Σ_s' P(s'|s,a) V(s')
    """
    S, A, P, R, gamma = MDP
    n = len(S)
    m = len(A)
    state_idx = {s: i for i, s in enumerate(S)}
    Q = np.zeros((n, m))

    for i, s in enumerate(S):
        for j, a in enumerate(A):
            r_sa = R.get(f"{s}-{a}", 0)
            sum_PV = 0
            for s_prime in S:
                key_p = f"{s}-{a}-{s_prime}"
                if key_p in P:
                    k = state_idx[s_prime]
                    sum_PV += P[key_p] * V[k].item()
            Q[i, j] = r_sa + gamma * sum_PV

    return Q


# 将 MDP 在策略1下转化为 MRP，并计算状态价值函数
P_mrp, R_mrp = marginalize_mdp_to_mrp(MDP, Pi_1)
print("策略1下的马尔可夫奖励过程 MRP 的转移矩阵 P:\n", P_mrp)
print("\n策略1下的马尔可夫奖励过程 MRP 的奖励向量 R:\n", R_mrp)
V = compute_value(P_mrp, R_mrp, gamma)
print("\n策略1下的状态价值函数 V:\n", V)

# 计算并打印动作价值函数
Q = compute_action_value(MDP, V)
print("\n策略1下的动作价值函数 Q:")
print(f"{'':>8}", end="")
for a in A:
    print(f"{a:>8}", end="")
print()
for i, s in enumerate(S):
    print(f"{s:>8}", end="")
    for j in range(len(A)):
        print(f"{Q[i, j]:>8.4f}", end="")
    print()



# 蒙特卡洛方法
def sample(MDP, Pi, timestep_max, number):
    '''采样函数，策略Pi， 限制最长时间步timestep_max, 采样序列数number'''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(len(S) - 1)]
        while s != S[-1] and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(s + '-' + a_opt, 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(s + '-' + a, 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(s + "-" + a + "-" + s_opt, 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


timestep_max = 20
episodes = sample(MDP, Pi_1, timestep_max, 1000)
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值函数 V\n", V)

# 估计占用度量
def occupancy(episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2)