import gymnasium as gym
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from policy_iteration import print_agent


'''
` ` ` `
` H ` H
` ` ` H
H ` ` G
'''
env = gym.make("FrozenLake-v1")
# 解封装才能访问状态转移矩阵P
env = env.unwrapped
# 环境渲染,通常是弹窗显示或打印出可视化的环境
env.render()

holes = set()
ends = set()

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[3] == True:       # 终止状态
                if s_[2] == 1.0:    # 奖励=1 → 目标
                    ends.add(s_[1])
                else:               # 奖励=0 且 done → 冰洞
                    holes.add(s_[1])
# 处理"终点自转移奖励为 0"的边界情况
holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)
for a in env.P[14]:  # 查看目标左边一格的状态转移信息
    print(env.P[14][a])


action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent1 = PolicyIteration(env, theta, gamma)
agent1.policy_iteration()
print_agent(agent1, action_meaning, holes, ends)

agent2 = ValueIteration(env, theta, gamma)
agent2.value_iteration()
print_agent(agent2, action_meaning, holes, ends)