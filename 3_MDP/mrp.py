import numpy as np
np.random.seed(0)

# 定义一个状态转移概率矩阵
P = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

# 定义奖励函数
rewards = np.array([-1, -2, -2, 10, 1, 0]) 
# 定义折扣因子
gamma = 0.5

def compute_return(chain, gamma):
    """
    计算经过状态链chain的折扣回报
    :param chain: 状态链, 如s1-s2-s3-s6，记录为 [1, 2, 3, 6]
    :param gamma: 折扣因子
    :return: 折扣回报
    """
    G = 0
    for t, state in enumerate(chain):
        G += (gamma ** t) * rewards[state - 1]  # state - 1 因为状态是从1开始的，而数组索引是从0开始的
    return G


def compute_value(P, rewards, gamma):
    """
    使用贝尔曼方程迭代计算价值函数
    :param P: 状态转移概率矩阵
    :param rewards: 奖励函数
    :param gamma: 折扣因子
    :return: 价值函数V
    """
    states_num = P.shape[0]
    # 将奖励函数转换为列向量
    rewards = np.array(rewards).reshape(-1, 1)
    value = np.dot(np.linalg.inv(np.eye(states_num) - gamma * P), rewards)
    return value
