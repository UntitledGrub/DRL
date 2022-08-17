import gym
import time
import pathlib
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')

env = gym.wrappers.AtariPreprocessing(env, noop_max=30, 
                                        frame_skip=4, 
                                        screen_size=84, 
                                        terminal_on_life_loss=False, 
                                        grayscale_obs=True, 
                                        grayscale_newaxis=False, 
                                        scale_obs=False)

env = gym.wrappers.FrameStack(env, 4)

env = gym.wrappers.RecordVideo(env, pathlib.Path(__file__).parent.resolve())

class Qnetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # flattens the output for the fully connected layer
        x = x.view(x.size(0), -1) 

        x = F.leaky_relu(self.fc1(x))

        x = F.leaky_relu(self.fc2(x))

        return x

class Qagent():

    def __init__(self, greed=0.2, greed_decay=0.95, discount_factor=0.1, lr=0.0003):
        self.memory = deque(maxlen=5000)

        self.greed = greed
        self.greed_decay = greed_decay
        self.discount = discount_factor

        self.lr = lr

        self.policy_network = Qnetwork()

    def choose_action(self, state):
        if np.random.rand() < self.greed: 
            return env.action_space.sample()

        action_values = self.policy_network(state)[0]

        return torch.argmax(action_values).item()

champ = Qagent()

EPISODES = 10
TIMESTEPS = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fill in path for the model you want to use
champ.policy_network.load_state_dict(torch.load('15000_policynetwork.pt'))
champ.policy_network.eval()

champ.policy_network.to(DEVICE)

start = time.time()
for e in range(1, EPISODES+1):
    state = np.array(env.reset())
    env.start_video_recorder()
    total_reward = 0
    for t in range(TIMESTEPS):
        action = champ.choose_action(torch.tensor(state, dtype=torch.float32, device=DEVICE).view(1, 4, 84, 84))
        _, reward, done, _ = env.step(action)
        total_reward += reward

        if done == True: 
            print(f"Episode {e}/{EPISODES} ended after {t} timesteps\tScore: {total_reward}\tTime elapsed: {round(time.time()-start, 3)}")
            break 
   
    env.close()


