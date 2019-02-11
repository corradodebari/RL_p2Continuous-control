# Continuous Control Project Report

## Learning Algorithm

The Learning Algorith is based on the the [Deep Determanistic Policy Gradient (DDPG) implementation](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py). The actor and critic networks are implemented as follow:

Actor:
- Input layer: 33
- fully connected layer: units=128
- activation: ReLU
- batch normalization layer
- fully connected layer: units=128 
- activation: ReLU
- fully connected: 4
- Output activation: tanh()

Critic:
- Input layer: 33
- fully connected layer: units=128
- activation: ReLU
- batch normalization layer
- fully connected layer: units=128 + 4 (actions)
- activation: ReLU
- Output fully connected: 1

Training optimizer used it's an Adam with a learning rate of 2e-4 for both networks, with a weight decay of 0 and a batch size of 128.
For the environment has been used a single Agent, that reached the target average score of 30 over 100 consecutive episodes after 347 episodes!
The progress chart has been:
<p align="left">
  <img src="running.png" width="350" title="hover text">
</p>

The Agent, before training, runs as in this video:

<p align="left">
  <img src="arm-not-training.gif" width="400" title="hover text">
</p>

After Training, the agent runs as follows:

<p align="left">
  <img src="arm-trained.gif" width="400" title="hover text">
</p>


## Further Improvement

Several improvements might be done to increase the network perfomances, as:
* reduce variance
* use 20 agents in parallel to exploit better DDPG algorithm
* try batch normalization on each hidden layer
* larger hidden layers
