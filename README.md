# Deep Reinforcement Learning 
Reimplementation of deep reinforcement learning for playing Atari 2600 games using experience replay. Original paper: https://www.nature.com/articles/nature14236

## Dependencies
torch <br />
gym <br />
NumPy <br />

## Implementation

Neural networks are typically trained by comparing the network's outputs to ground truths. In this case ground truths would be the optimal action for each state, where a state is a stack of 4 frames from the game. These ground truths are not available. Replay memory addresses this problem. Experiences are stored as tuples e = (state, action, reward, next_state). Initially the agent plays randomly to generate a stock of experiences. 

During training the model considers a batch of experiences and generates a predicted value for the action taken in the state in each experience. This prediction is then compared to the sum of the actual reward gained by taking that action in that state, and the predicted value of the highest valued action given next_state. This does mean that the model is chasing its own tail to some extent, since it updates its weights based on a prediction about the next state. That said, the actual reward from the experience stored in replay memory gives the model some grounding in actual experience to base parameter updates on. 

## Training
To train the agent run the script q_agent_train.py. By default the agent plays 15,000 training games, for better results increase the number of training games.

The agent learns to play Breakout, but any Atari 2600 game available in gym is compatible. To train with another game replace 'BreakoutNoFrameSip-v4' in the line shown below with the name of a different game supported by gym (make sure to use the no frame skip version of your chosen game): <br />
![image](https://user-images.githubusercontent.com/34168073/188141947-8db1eb60-d5fd-4c1a-a07a-5f26c1cb7908.png)

The state of the model is updated after each time step of each training game and is saved every 50 training games. 

## Creating videos of your agent's performance
The script trained_model_video_generation.py displays and saves videos of your model's performance. 

Run the script from the command line and append the argument --state-dict trained_model_state_dict.pt. 

## Results
For human-level performance, training a Q-learning agent takes a very long time. The results below show consistent improvement over 15,000 training games, but this took over 30 hours to run on an NVidia RTX 2060 and the model did not apporach human-level performance during training. We also tried training the agent to play Pacman, which has a much larger state space than Breakout, and the agent made little progress with Pacman after 15,000 training games.

![markfigure](https://user-images.githubusercontent.com/34168073/188142711-98517dc2-a1e1-4282-ab64-225a8d72ba1b.png)

The agent steadily imporves over the first 9,000 training games, but learning becomes more erratic thereafter. 
