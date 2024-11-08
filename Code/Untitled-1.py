
def get_state(row,col):
    if row != 3:
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    if row == 3 and col == 11:
        return 'terminal'
    return 'trap'

def move(row,col,action):
	# 如果已经在悬崖，或者重点，反馈就是0
	if get_state(row,col) in ['trap','terminal']:
		return row,col,0
	# 👆
	if action == 0:
		row -= 1
	# 👇
	if action == 1:
		row += 1
	# 👈
	if action == 2:
		col -= 1
	# 👉
	if action == 3:
		col += 1
	#避免走到地图外面
	row = max(0,row)
	row = min(3,row)
	col = max(0,col)
	col = min(11,col)
	#如果是悬崖，奖励就是-100，否则都是-1
	reward = -1
	if get_state(row,col) == 'trap':
		reward = -100
	return row,col,reward

import numpy as np
#初始化Value和pi
values = np.zeros([4,12])
pi = np.ones([4,12,4])*0.25

# 动作价值函数
# 计算一个状态下执行动作的分数
def get_qsa(row,col,action):
    # 计算下一个状态的价值
    next_row , next_col,reward = move(row,col,action)
    # 计算下一个状态的分数，0.9是折扣因子
    value = values[next_row,next_col] * 0.9
    # 如果是悬崖或者陷阱，下一个状态分数是0
    if get_state(next_row,next_col) in ['trap','terminal']:
        value = 0

    # 更新当前状态的价值，是把之前的和现在的加起来
    return value + reward


# %% [markdown]
# ## 策略评估
# 
# 评估每一个格子的价值

# %%
def get_values():

    # 初始化一个新的values,重新评估所有格子的分数

    new_values = np.zeros([4,12])

   # 遍历所有格子 
    for row in range(4):
        for col in range(12):
            
            # 计算当前格子4个动作分别的分数
            action_value = np.zeros(4)

            # 遍历动作
            for action in range(4):
                action_value[action] = get_qsa(row, col, action)
            
            # 计算当前格子的分数
            action_value *= pi[row, col]

            # 所有动作分数求和
            new_values[row, col] = action_value.sum()

    return new_values



def get_pi():
    # 重新初始化每个格子下采用动作的概率，重新评估
    new_pi = np.zeros([4,12,4])
    # 遍历
    for row in range(4):
        for col in range(12):
            
            action_value = np.zeros(4)

            for action in range(4):
                action_value[action] = get_qsa(row, col, action)
            
            # 计算每个动作的概率，根据达到最大值均分
            count = (action_value == action_value.max()).sum()

            for action in range(4):
                if action_value[action] == action_value.max():
                    new_pi[row,col,action] = 1/count
                else:
                    new_pi[row,col,action] = 0

    return new_pi

# %%
for i in range(10):
    for j in range(10):
        values = get_values()
    pi = get_pi()


# %%
for row in range(4):
    line = ""
    for col in range(12):
        action = pi[row,col].argmax()
        action = {0: "👆", 1: "👇", 2: "👈", 3: "👉"}[action]
        line += action
    print(line)


