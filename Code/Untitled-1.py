# %%
import gym
from matplotlib import pyplot as plt
# %matplotlib inline
import os

# %%
os.environ['SDL_VIDEODRIVER'] = 'dummy'
env = gym.make('FrozenLake-v1',
               is_slippery=False,
               map_name='4x4',
               desc=['SFFF', 'FHFH', 'FFFH', 'HFFG'],
               render_mode='rgb_array'
               )
env.reset()

# %%
env = env.unwrapped

# %%
def show():
    screen = env.render()
    plt.imshow(screen)
    # plt.imshow(env.render())
    plt.axis('off')
    plt.show()

# %%
show()

# %%
len(env.P)
env.P[2]

# %%
env.P[0][2]

# %%
# 初始化
import numpy as np
values = np.zeros(16)
pi = np.ones([16,4])*0.25

algorithm = "value iteration" #价值迭代
algorithm = "policy iteration" #策略迭代

# %%
# 计算qsa
def get_qsa(state,action):
    value = 0.0

    for prop, next_state, reward, over in env.P[state][action]:
        
        # 计算下一个状态的价值
        next_value = values[next_state] *0.9
        # 如果下一个状态是终止状态，则价值为0
        if over:
            next_value = 0.0

        # 奖励加状态等于最终分数
        next_value += reward

        # 状态转移概率
        next_value *= prop

        value += next_value

    return value

# %%
# 策略评估
def get_values():
    new_values = np.zeros(16)
    for state in range(16):

        action_value = np.zeros(4)
        for action in range(4):
            action_value[action] = get_qsa(state, action)

        if algorithm == 'policy iteration':
            action_value *= pi[state]
            new_values[state] = action_value.sum()
        if algorithm =='value iteration':
            new_values[state] = action_value.max()

    return new_values

# %%
# 策略提升

def get_pi():
    
    new_pi = np.zeros([16,4])

    for state in range(16):

        action_value = np.zeros(4)

        for action in range(4):
            action_value[action] = get_qsa(state, action)

        count = (action_value == action_value.max()).sum()

        for action in range(4):
            if action_value[action] == action_value.max():
                new_pi[state][action] = 1/count
            else:
                new_pi[state][action] = 0
    
    return new_pi

# %%
for _ in range(10):
    for _ in range(100):
        values = get_values()
    pi = get_pi()


# %%
# 查看策略

def print_pi():
    for row in range(4):
        line = ""
        for col in range(4):
            state = row * 4 + col

            if (row == 1 and col == 1) or (row == 1 and col == 3) or(row == 2 and col == 3) or (row == 3 and col == 0) :
                line += "⚪"
                continue

            if row == 3 and col == 3:
                line += "♥"
                continue

            line += '←↓→↑'[pi[state].argmax()]
        print(line)

# %%
print_pi()

# %%
print(pi)

# %%
import numpy as np
import matplotlib.pyplot as plt

# 假设 pi 为您提供的概率数组
pi = np.array([[0.   , 0.5  , 0.5  , 0.  ],
                [0.   , 0.   , 1.   , 0.  ],
                [0.   , 1.   , 0.   , 0.  ],
                [1.   , 0.   , 0.   , 0.  ],
                [0.   , 1.   , 0.   , 0.  ],
                [0.25 , 0.25 , 0.25 , 0.25],
                [0.   , 1.   , 0.   , 0.  ],
                [0.25 , 0.25 , 0.25 , 0.25],
                [0.   , 0.   , 1.   , 0.  ],
                [0.   , 0.5  , 0.5  , 0.  ],
                [0.   , 1.   , 0.   , 0.  ],
                [0.25 , 0.25 , 0.25 , 0.25],
                [0.25 , 0.25 , 0.25 , 0.25],
                [0.   , 0.   , 1.   , 0.  ],
                [0.   , 0.   , 1.   , 0.  ],
                [0.25 , 0.25 , 0.25 , 0.25]])

# 绘制热图
plt.figure(figsize=(10, 6))
plt.imshow(pi, cmap='Blues', aspect='auto')
plt.colorbar(label='Probability')
plt.title('Action Probability Distribution (pi)')
plt.xlabel('Actions (0: Left, 1: Down, 2: Right, 3: Up)')
plt.ylabel('States')
plt.xticks(ticks=np.arange(4), labels=['Left', 'Down', 'Right', 'Up'])
plt.yticks(ticks=np.arange(len(pi)), labels=np.arange(len(pi)))
plt.grid(False)
plt.show()


# %%
from IPython import display

import time

def play():
    env.reset()
    index = 0

    for i in range(200):
        action = np.random.choice(np.arange(4),size=1,p=pi[index])[0]

        index, reward, teminated,truncated,_= env.step(action)

        display.clear_output(wait=True)
        time.sleep(0.1)
        show()
        if teminated or truncated:
            break

    print(i)

# %%
play()


