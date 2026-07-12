class CliffWalkingEnv:
    '''
        网格布局，左上角坐标为（0， 0）
        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        .  .  .  .  .  .  .  .  .  .  .  .
        S  C  C  C  C  C  C  C  C  C  C  G
    '''
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        # 记录当前agent的位置坐标
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        # 当以左上角为原点时，“上”动作对应纵坐标-1；“下”动作对应纵坐标+1
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.ncol * self.y + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self. ncol + self.x
