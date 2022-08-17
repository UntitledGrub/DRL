import gym
import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import copy

env = gym.make('BreakoutNoFrameskip-v4')

# Puts the frames of the game in grayscale
# Sets a limit for how many null actions the agent can take (30)
# Set the size of the frames, 84 x 84
# Sets the frequency with which the agent takes actions/the gap between observations, once every 4 frame
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, 
                                        frame_skip=4, 
                                        screen_size=84, 
                                        terminal_on_life_loss=False, 
                                        grayscale_obs=True, 
                                        grayscale_newaxis=False, 
                                        scale_obs=False)

# Transforms the frames into stacks of 4 to give the agent context
env = gym.wrappers.FrameStack(env, 4)

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

    def __init__(self, epsilon=1, epsilon_decay=0.995, discount_factor=0.95, lr=0.0003):
        self.memory = deque(maxlen=10000)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.discount = discount_factor

        self.epsilon_min = 0.01

        self.lr = lr

        self.policy_network = Qnetwork()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon: 
            return env.action_space.sample()

        action_values = self.policy_network(state)[0]

        return torch.argmax(action_values).item()

#### Training the Deep Q-Learning Agent ####

champ = Qagent(epsilon=1, epsilon_decay=0.995, discount_factor=0.95, lr=0.0003)

EPISODES = 15000
TIMESTEPS = 10000
BATCHSIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

champ.policy_network.to(DEVICE)
opt = optim.Adam(champ.policy_network.parameters(), lr=champ.lr)
loss_function = nn.MSELoss(reduction='mean')

target_network = copy.deepcopy(champ.policy_network)

data = dict()

champ.policy_network.train()

start = time.time()
for e in range(1, EPISODES+1):
    
    state = np.array(env.reset())
    total_reward = 0
    for t in range(TIMESTEPS):
        action = champ.choose_action(torch.tensor(state, dtype=torch.float32, device=DEVICE).view(1, 4, 84, 84))
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state)

        total_reward += reward

        # We check whether the state was a terminal one before we choose to commit it to memory
        # This is to avoid recording terminal states to memory
        if done == True: 
            print(f"Episode {e}/{EPISODES} ended after {t} timesteps\tScore: {total_reward}\tTime elapsed: {round(time.time()-start, 3)}")
            data[e] = [t, total_reward, round(time.time()-start, 3)]
            break

        # Only if next_state is not terminal do we add it to champ's replay memory
        champ.memory.append([torch.tensor(state, dtype=torch.float32, device=DEVICE).view(1, 4, 84, 84), action, reward, torch.tensor(next_state, dtype=torch.float32, device=DEVICE).view(1, 4, 84, 84)])
        
        state = next_state

        if len(champ.memory) >= BATCHSIZE:
            if t % 100 == 0:
                target_network = copy.deepcopy(champ.policy_network)

            batch = random.sample(champ.memory, BATCHSIZE)

            # Concatenates the tensors representing states so they can be processed in one forward pass of the Q network
            states = [experience[0] for experience in batch]
            states = torch.cat(states)
            states.to(DEVICE)

            # Concatenates the tensors representing next states so they can be processed in one forward pass of the Q network
            next_states = [experience[3] for experience in batch]
            next_states = torch.cat(next_states)
            next_states.to(DEVICE)

            # The actions are formatted into a tensor to index into the output of the policy network
            # The policy network outputs a vectors of action value estimates
            # We need the action indices to get the action value estimates for the actions contained in the experiences in the current training batch
            actions = torch.tensor([experience[1] for experience in batch], device=DEVICE)
            actions = actions.view(32, 1)

            # The rewards are put into a tensor so they can be added to the output of the target network
            # This is one of the ingredients we need to compute the target values, i.e. the values of the Bellman optimality equation
            rewards = torch.tensor([experience[2] for experience in batch], device=DEVICE)

            actual = torch.gather(champ.policy_network(states), 1, actions)
            
            next_state_q_values = torch.amax(target_network(next_states), dim=1)

            target = rewards + champ.discount * next_state_q_values

            opt.zero_grad()

            loss = loss_function(actual.flatten(), target)
            loss.backward()

            opt.step()

    if e % 50 == 0:
        # We do not allow epsilon to decay below a preset minimum
        champ.epsilon = champ.epsilon*champ.epsilon_decay if champ.epsilon*champ.epsilon_decay >= champ.epsilon_min else champ.epsilon
        torch.save(champ.policy_network.state_dict(), f'{e}_policynetwork.pt')

    env.close()

file = open(f"results.csv", 'a')
for ep, [timesteps, reward, execution_time] in data.items():
    file.write(f"{ep},{timesteps},{reward},{execution_time}\n")
file.close()
    
print(f"Time to execute {EPISODES} episodes: {round(time.time() - start, 3)}")
