# Deep Reinforcement Learning 
Reimplementation of deep reinforcement learning for playing Atari 2600 games using experience replay. Original paper: https://www.nature.com/articles/nature14236

## Dependencies
torch <br />
gym <br />
NumPy <br />

## Training
To train the agent run the script q_agent_train.py. By default the agent plays 15,000 training games, for better results increase the number of training games.

The agent learns to play Breakout, but any Atari 2600 game available in gym is compatible. To train with another game replace 'BreakoutNoFrameSip-v4' with the name of a different game supported by gym:
![image](https://user-images.githubusercontent.com/34168073/188141947-8db1eb60-d5fd-4c1a-a07a-5f26c1cb7908.png)

The state of the model is updated after time step of each training game and is saved every 50 training games. 

Training a Q-learning agent takes a very long time, 

## Creating videos of your agent's performance
The script trained_model_video_generation.py displays and saves videos of your model's performance. 

Run the script from the command line and append the argument --state-dict trained_model_state_dict.pt. 

## Results
![markfigure](https://user-images.githubusercontent.com/34168073/188142711-98517dc2-a1e1-4282-ab64-225a8d72ba1b.png)

The agent steadily imporves over the first 9,000 training games, but learning becomes more erratic thereafter. 
