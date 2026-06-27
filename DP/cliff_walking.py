'''悬崖漫步问题'''
import copy

class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP()

    def createP(self):
        '''
        网格布局，左上角坐标为（0， 0）
        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        S  C  C  C  C  C  C  C  C  C  C  G
        '''
        # (转移概率， 位置，奖励，是否结束)
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]

        # 定义移动
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    else:
                        next_j = min(self.ncol - 1, max(0, j + change[a][0]))
                        next_i = min(self.nrow - 1, max(0, i + change[a][1]))
                        next_state = next_i * self.ncol + next_j
                        reward = -1
                        done = False
                        if next_i == self.nrow - 1 and next_j > 0:
                            done = True
                            if next_j != self.ncol - 1:
                                reward = -100
                            
                        P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P
